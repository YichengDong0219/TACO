
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import set_seed
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import time


from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

import logging
from pprint import pformat
import json
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


def evaluate(model, val_dataset, device, cfn_action_steps):
    # model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=device.type != "cpu"
    )
    total_val_loss = 0.0
    total_val_norm = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:

            loss, model_output_val = model.compute_loss(batch)
            total_val_loss += loss.item()
            
            val_norm = model_output_val.norm(p=2, dim=1).mean()  
            total_val_norm += val_norm.item()

            num_batches += 1

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_norm = total_val_norm / num_batches 
    return avg_val_loss, avg_val_norm

def the_str_have(a_str, a_str_list):
    for ystr in a_str_list:
        if ystr in a_str.lower():
            pass
        else:
            return False
    return True

def check_task_is_simpler_bridge(the_str):
    if (
        the_str_have(the_str, ["put ", " on ", "carrot"]) or
        the_str_have(the_str, ["put ", " on ", "spoon"]) or
        the_str_have(the_str, ["stack ", " on "]) or
        the_str_have(the_str, ["put ", "eggplant"])
    ):
        return True
    return False


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    overall_start = time.time()
    # breakpoint()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    t0 = time.time()
    dataset = make_dataset(cfg)
    t1 = time.time()
    print(f"📦 dataset loading time: {t1 - t0:.2f}s")

    torch.set_printoptions(sci_mode=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    pretrained_checkpoint_path="your/pretrained/policy/path"
    print(f"loading pretrained checkpoint from {pretrained_checkpoint_path}...")

    policy=PI05Policy.from_pretrained(
        pretrained_name_or_path=pretrained_checkpoint_path, 
        local_files_only=True
    )
    policy.eval()    

    print("loading model success!")
    for p in policy.parameters():
        p.requires_grad = False
    

    policy_config = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=pretrained_checkpoint_path, 
        local_files_only=True
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=pretrained_checkpoint_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        # preprocessor_overrides={"device_processor": {"device": str(policy_config.device)}},
    )
    

    num_batches_per_epoch = len(dataset) // cfg.batch_size
    total_epochs = 16  # cfg.steps // num_batches_per_epoch + 1

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    seed = 42
    torch.manual_seed(seed)
    print(f"noise seed is {seed} !!!")

    noise_num = 1
    sample_num = 50
    assert policy.model.config.n_action_steps == 20
    actions_shape = (sample_num, policy.model.config.n_action_steps, policy.model.config.max_action_dim)
    noise42 = torch.normal(
        mean=0.0,
        std=1.0,
        size=actions_shape,
        dtype=dtype,
    ).to(device)
    print(f"noise is\n{noise42}")

    noise42 = noise42.unsqueeze(1).repeat(1, cfg.batch_size, 1, 1).reshape(sample_num*cfg.batch_size, policy.model.config.n_action_steps, policy.model.config.max_action_dim)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir is {output_dir} !!!")

    total_epochs = 1
    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} start")
        # optimizer.zero_grad()

        save_path = output_dir
        batch_num = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", maxinterval=0.5)
        for batch in pbar:
            batch_num += 1
            batch['observation.state'] = batch['observation.state'].to(device)
            batch['observation.images.cam_0'] = batch['observation.images.cam_0'].to(device)
            batch['observation.images.cam_1'] = batch['observation.images.cam_1'].to(device)
            batch['observation.images.cam_2'] = batch['observation.images.cam_2'].to(device)

            batch['action'] = batch['action'].to(device)
            bs = batch['observation.state'].shape[0]

            batch['observation.state'] = batch['observation.state'].unsqueeze(0)
            batch['observation.images.cam_0'] = batch['observation.images.cam_0'].unsqueeze(0)
            batch['observation.images.cam_1'] = batch['observation.images.cam_1'].unsqueeze(0)
            batch['observation.images.cam_2'] = batch['observation.images.cam_2'].unsqueeze(0)
            batch['action'] = batch['action'].unsqueeze(0)

            batch['observation.state'] = batch['observation.state'].repeat(noise_num, 1, 1).reshape(noise_num*bs, 8)
            batch['observation.images.cam_0'] = batch['observation.images.cam_0'].repeat(noise_num, 1, 1, 1, 1).reshape(noise_num*bs, 3, 480, 640)
            batch['observation.images.cam_1'] = batch['observation.images.cam_1'].repeat(noise_num, 1, 1, 1, 1).reshape(noise_num*bs, 3, 480, 640)
            batch['observation.images.cam_2'] = batch['observation.images.cam_2'].repeat(noise_num, 1, 1, 1, 1).reshape(noise_num*bs, 3, 480, 640)
            batch['task'] = batch['task'] * noise_num

            features_good = []
            with torch.no_grad():
                batch = preprocessor(batch)
                nor_actions, features = policy.predict_action_chunk_and_get_feature_opt(sample_num, batch, noise42.clone())
                # import ipdb;ipdb.set_trace()
                # batch = policy.normalize_targets(batch)
                gt_action = batch['action'].repeat(sample_num, 1, 1, 1)
                # gt_action = batch['action'].reshape(sample_num, bs, 20, 16)
                nor_actions = nor_actions.reshape(sample_num, bs, 20, 8)
                features = features.reshape(sample_num, bs, 1024)
                # import ipdb;ipdb.set_trace()

                dis = torch.norm(nor_actions - gt_action, dim=(2, 3), p=2)
                for j in range(bs):
                    min_index = torch.argmin(dis[:, j])
                    features_good.append(features[min_index, j, :].cpu())
                    # import ipdb;ipdb.set_trace()
                    # print()
                torch.save(features_good, save_path / f"feature{batch_num}.pt")

        print(f"features have been saved at {save_path / 'featurei.pt'}")
        print()


if __name__ == "__main__":
    train()
