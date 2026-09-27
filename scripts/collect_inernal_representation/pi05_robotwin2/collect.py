
import os
import logging
from pprint import pformat
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from tqdm import tqdm
from cfn.pi05_rpc import PI05RPCClient


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    policy_server = os.getenv("TACO_POLICY_SERVER")
    # A remote client keeps dataset decoding on CPU; the server owns the GPU.
    device = torch.device("cpu") if policy_server else get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Creating policy client" if policy_server else "Creating policy")
    pretrained_checkpoint_path = cfg.policy.pretrained_path
    if policy_server:
        client = PI05RPCClient(policy_server)
        client.health()
        policy = None
        preprocessor = None
    else:
        policy=PI05Policy.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path,
            local_files_only=True
        )
        policy.eval()
        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path,
            local_files_only=True
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=pretrained_checkpoint_path,
        )

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    dtype = next(policy.parameters()).dtype if policy is not None else torch.float32
    seed = 42
    torch.manual_seed(seed)
    print(f"noise seed is {seed} !!!")
    noise_num = 50
    if policy is not None:
        actions_shape = (noise_num, policy.model.config.n_action_steps, policy.model.config.max_action_dim)
        noise42 = torch.normal(
            mean=0.0,
            std=1.0,
            size=actions_shape,
            dtype=dtype,
        ).to(device)
        print(f"noise is\n{noise42}")

    if policy is not None:
        policy.eval()
    features_good = []

    logging.info("Start selecting noise")
    for batch in tqdm(dataloader):
        if policy is not None:
            batch['observation.state'] = batch['observation.state'].to(device)
            batch['observation.images.cam_high'] = batch['observation.images.cam_high'].to(device)
            batch['observation.images.cam_left_wrist'] = batch['observation.images.cam_left_wrist'].to(device)
            batch['observation.images.cam_right_wrist'] = batch['observation.images.cam_right_wrist'].to(device)
            batch['action'] = batch['action'].to(device)
        bs = batch['observation.state'].shape[0]

        for i in tqdm(range(bs)):
            noise_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    noise_batch[key] = value[[i]]
                elif isinstance(value, list):
                    noise_batch[key] = [value[i]]
                else:
                    assert 0, "this batch is not right"

            if policy_server:
                feature = client.request(
                    "feature",
                    state=noise_batch['observation.state'][0].numpy(),
                    images=[
                        noise_batch['observation.images.cam_high'][0].numpy(),
                        noise_batch['observation.images.cam_left_wrist'][0].numpy(),
                        noise_batch['observation.images.cam_right_wrist'][0].numpy(),
                    ],
                    action=noise_batch['action'][0].numpy(),
                    instruction=noise_batch['task'][0],
                )
                features_good.append(torch.from_numpy(feature))
                continue

            noise_batch['observation.state'] = noise_batch['observation.state'].repeat(noise_num, 1)
            noise_batch['observation.images.cam_high'] = noise_batch['observation.images.cam_high'].repeat(noise_num, 1, 1, 1)
            noise_batch['observation.images.cam_left_wrist'] = noise_batch['observation.images.cam_left_wrist'].repeat(noise_num, 1, 1, 1)
            noise_batch['observation.images.cam_right_wrist'] = noise_batch['observation.images.cam_right_wrist'].repeat(noise_num, 1, 1, 1)
            noise_batch['action'] = noise_batch['action'].repeat(noise_num, 1, 1)
            noise_batch['task'] = noise_batch['task'] * noise_num

            with torch.no_grad(): 
                noise_batch = preprocessor(noise_batch)
                gt_action = noise_batch['action']
                actions, features = policy.predict_action_chunk_and_get_feature(noise_batch, noise42.clone())

                # select best noise
                norm42 = torch.norm(actions - gt_action, dim=(1, 2), p=2)
                min_index = torch.argmin(norm42)
                features_good.append(features[min_index])
    
    features_good = torch.stack(features_good)
    save_path = cfg.output_dir
    os.makedirs(save_path, exist_ok=True)
    torch.save(features_good, save_path / "feature.pt")
    print(f"features have been saved at {save_path / 'feature.pt'}")
    print()


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
