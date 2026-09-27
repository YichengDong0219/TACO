#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import sys
import numpy as np

import json
import numpy as np
import torch

from lerobot.policies.pi05 import PI05Policy

from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

from cfn.cfn_net import CFN

class Lerobot_torch_PI05:
    def __init__(self, task_name, pretrained_checkpoint_path):
        self.task_name = task_name
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        model=PI05Policy.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path, 
            local_files_only=True
        )
        model.eval()

        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path, 
            local_files_only=True
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=pretrained_checkpoint_path,
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor


        self.policy = model # model.policy

        self.img_size = (224,224)
        self.observation_window = None

        self.action_mask = torch.ones(14, dtype=torch.bool).cuda()
        self.action_mask[13] = False
        self.action_mask[6] = False

        self.num_result = 1
        self.pi0_step = 50

        # init norm log
        self.test_num=1
        self.norm_dict = {
            f"test{self.test_num}": {
                "norms_means": [],
                "norms_selected": [],
                "mean": -1,
                "is_suc": -1,
            }
        }


    def logn(self, norms, norm_min):
        self.norm_dict[f"test{self.test_num}"]["norms_means"] += [norms.mean().item()]
        self.norm_dict[f"test{self.test_num}"]["norms_selected"] += [norm_min.item()]

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        img_front = np.transpose(img_front, (2, 0, 1))  
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        img_front = torch.from_numpy(img_front / 255.)
        img_left = torch.from_numpy(img_left / 255.)
        img_right = torch.from_numpy(img_right / 255.)

        para = next(self.policy.parameters())
        dtype = para.dtype

        self.observation_window = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(dtype),
            "observation.images.cam_high": img_front.unsqueeze(0).to(dtype),
            "observation.images.cam_left_wrist": img_left.unsqueeze(0).to(dtype),
            "observation.images.cam_right_wrist": img_right.unsqueeze(0).to(dtype),
            "task": [self.instruction],
        }

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        with torch.no_grad():
            inputs = {k: v.to(self.policy.config.device) if isinstance(v, torch.Tensor) else v for k, v in self.observation_window.items()}
                    
            for key, value in inputs.items():
                if key == 'task':
                    assert len(inputs[key]) == 1
                    inputs[key] = inputs[key] * self.num_result
                elif len(inputs[key].shape) == 4:
                    inputs[key] = inputs[key].repeat(self.num_result, 1, 1, 1)
                elif len(inputs[key].shape) == 2:
                    inputs[key] = inputs[key].repeat(self.num_result, 1)
                else:
                    print("error in inputs\n")
                    assert 0
            
            inputs = self.preprocessor(inputs)
            actions = self.policy.predict_action_chunk(inputs)
            actions_un = self.postprocessor(actions)
            
        return actions_un[0].to(torch.float32).cpu().numpy()

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.test_num += 1
        self.norm_dict[f"test{self.test_num}"] = {
                "norms_means": [],
                "norms_selected": [],
                "mean": -1,
                "is_suc": -1,
            }
        print("successfully unset obs and language intruction")

class Lerobot_torch_PI05_taco:
    def __init__(
        self, 
        task_name, 
        pretrained_checkpoint_path, 
        cfn_ckpt_path,
    ):
        self.task_name = task_name
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        model=PI05Policy.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path, 
            local_files_only=True
        )
        model.eval()

        policy_config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_checkpoint_path, 
            local_files_only=True
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=pretrained_checkpoint_path,
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.policy = model # model.policy

        cfn = CFN(
            cfn_output_dim=20,
            cfn_hidden_dim=1536
        ).to(next(model.parameters()).device)
        weight_path = cfn_ckpt_path

        print(f"🔍 load cfn ckpt: {weight_path}")
        cfn.cfn.load_state_dict(
            torch.load(weight_path, map_location=next(model.parameters()).device, weights_only=True)
        )

        self.cfn = cfn.cfn.eval()


        self.img_size = (224,224)
        self.observation_window = None

        self.action_mask = torch.ones(14, dtype=torch.bool).cuda()
        self.action_mask[13] = False
        self.action_mask[6] = False

        self.num_result = 50
        self.pi0_step = 50

        # init norm log
        self.test_num=1
        self.norm_dict = {
            f"test{self.test_num}": {
                "norms_means": [],
                "norms_selected": [],
                "mean": -1,
                "is_suc": -1,
            }
        }

        # get noise
        para = next(self.policy.parameters())
        bsize = self.num_result
        device = para.device
        dtype = para.dtype
        actions_shape = (bsize, self.policy.model.config.n_action_steps, self.policy.model.config.max_action_dim)
        seed = 42
        torch.manual_seed(seed)
        print(f"noise seed is {seed} !!!")
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=actions_shape,
            dtype=torch.float32,
        ).to(device).to(dtype)
        self.noise = noise
        print(f"noise is\n{noise}")


    def logn(self, norms, norm_min):
        self.norm_dict[f"test{self.test_num}"]["norms_means"] += [norms.mean().item()]
        self.norm_dict[f"test{self.test_num}"]["norms_selected"] += [norm_min.item()]

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = img_arr[0], img_arr[1], img_arr[2], state
        img_front = np.transpose(img_front, (2, 0, 1))  
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        img_front = torch.from_numpy(img_front / 255.)
        img_left = torch.from_numpy(img_left / 255.)
        img_right = torch.from_numpy(img_right / 255.)

        para = next(self.policy.parameters())
        dtype = para.dtype

        self.observation_window = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(dtype),
            "observation.images.cam_high": img_front.unsqueeze(0).to(dtype),
            "observation.images.cam_left_wrist": img_left.unsqueeze(0).to(dtype),
            "observation.images.cam_right_wrist": img_right.unsqueeze(0).to(dtype),
            "task": [self.instruction],
        }

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        with torch.no_grad():
            inputs = {k: v.to(self.policy.config.device) if isinstance(v, torch.Tensor) else v for k, v in self.observation_window.items()}
                    
            for key, value in inputs.items():
                if key == 'task':
                    assert len(inputs[key]) == 1
                    inputs[key] = inputs[key] * self.num_result
                elif len(inputs[key].shape) == 4:
                    inputs[key] = inputs[key].repeat(self.num_result, 1, 1, 1)
                elif len(inputs[key].shape) == 2:
                    inputs[key] = inputs[key].repeat(self.num_result, 1)
                else:
                    print("error in inputs\n")
                    assert 0
            
            inputs = self.preprocessor(inputs)
            
            actions, features = self.policy.predict_action_chunk_and_get_feature(inputs, self.noise.clone())

            actions_un = self.postprocessor(actions)
            features = features.to(next(self.cfn.parameters()).dtype)
            cfn_output = self.cfn(features)

            norm = cfn_output.norm(dim=1)
            min_val = torch.min(norm)
            # self.logn(norm, min_val)
            indices = torch.nonzero(norm == min_val).squeeze()

            if indices.ndim == 0:
                selected_index = indices.item()
            else:
                selected_index = indices[torch.randint(0, len(indices), (1,))].item()    

            actions_un = actions_un[[selected_index], ...]
            
        return actions_un[0].to(torch.float32).cpu().numpy()

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.test_num += 1
        self.norm_dict[f"test{self.test_num}"] = {
                "norms_means": [],
                "norms_selected": [],
                "mean": -1,
                "is_suc": -1,
            }
        print("successfully unset obs and language intruction")
