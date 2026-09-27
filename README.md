# Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach

<div align="center">

<img src="assets/taco.png" alt="TACO" width="20"/> **Test-time Anti-exploration via pseudo-COunts (TACO)**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.02834)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://vla-anti-exploration.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Video-orange)](https://vla-anti-exploration.github.io/#:~:text=A%20comparison%20of%20key%20moments%20while%20grasping%20a%20marker)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/rhodes-team-teleai/vla-tts-taco)
</div>

---

## 📋 Overview
<p align="center">
  <img src="assets/overview.png" alt="TACO Architecture" width="800"/>
  <br>
  <em>TACO: Test-time Anti-exploration via pseudo-COunts</em>
</p>

**TACO** (**T**est-time **A**nti-exploration via pseudo-**CO**unts) is a novel test-time scaling framework for VLAs which retains the strong generalization capabilities of pretrained VLAs while effectively constraining outputs to the success modes of specific
downstream tasks, performing the **Anti-Exploration** principle in offline RL. By leveraging a lightweight **Coin Flipping Network (CFN)**, TACO acquires accurate measurement of distributional shift with minimal computational overhead, significantly improving performance on out-of-distribution testcases.

### 🎯 Key Features

- **Principled Anti-Exploration**: Mitigates inference-time fragility by constraining generated actions to the "success modes" of the downstream task, effectively handling distribution shifts.
- **Universal Compatibility**: Seamlessly integrates with **Flow-Matching** (e.g., $\pi_0$, $\pi_{0.5}$), **Diffusion** (e.g., RDT), and **Autoregressive** (e.g., OpenVLA) architectures.
- **Gradient-Free Steering**: Performs Test-Time Scaling (TTS) via a generate-then-verify pipeline without modifying the heavy VLA backbone parameters.
- **Efficient Inference**: Implements **KV Cache Optimization** to reuse visual-language representations, reducing inference latency by **~73%** compared to the original manner, during parallel sampling.
- **High-Fidelity Verification**: Utilizes a lightweight **Coin Flipping Network (CFN)** trained on internal representations with **High-Fidelity Feature Search** to accurately estimate action reliability.

### 🏆 Main Contributions

1. **New Perspective on VLA Instability**: We diagnose the inference fragility of generative VLAs as an out-of-support problem and propose **TACO**, the first framework to address this via the **Anti-Exploration** principle from Offline RL using Test-Time Scaling.
2. **Coupled Pseudo-Count Estimator**: We introduce an efficient internal representation mechanism coupled with a **High-Fidelity Feature Search** strategy. This allows the CFN to accurately verify action chunks for denoising-based policies (Flow/Diffusion) that never see clean actions during training.
3. **SOTA Performance**: Extensive experiments across **extensive simulation tasks** (RoboTwin, LIBERO, SimplerEnv) and **real-world dual-arm manipulation** demonstrate that TACO significantly boosts success rates (e.g., +16% in real-world tasks) over strong baselines like $\pi_0$.


## 📰 News

> - **[2025-12]** Releasing TACO code and models. See our [huggingface collections](https://huggingface.co/collections/rhodes-team-teleai/vla-tts-taco).

## 🛠️ Installation

### Install CFN

Create a conda env:
```
conda create -n taco python=3.10 -y
conda activate taco
```

Install torch (choose the version that suits your environment):
```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

Install CFN:
```
cd cfn/
pip install -e .
cd ..
```

### Install Pi0.5 (lerobot)

```
cd third_party/lerobot
pip install -e .

cd src/transformers
pip install -e .

cd ../../../..
```

#### (Optional) Install Robotwin2.0

This branch does not vendor RoboTwin. The multi-task RoboTwin pipeline clones
the official checkout and fetches its assets itself:

```
vendor/bootstrap.sh robotwin
```

See [ROBOTWIN12.md](./ROBOTWIN12.md) for the pinned revision and what the
pipeline expects of it. The upstream TACO release vendored an older RoboTwin
under `third_party/Robotwin` for its single-task eval path; that path was
removed here because it cannot run `convert` or `eval`. To get it back, see the
recovery command in ROBOTWIN12.md.

#### (Optional) Install Lerobot-Libero
```
cd ./third_party/lerobot
pip install -e ".[libero]"
```

### Install OpenVLA
You can create a new conda environment, refer [third_party/openvla](./third_party/openvla/README.md) to install Libero and our modified OpenVLA.

## 🚀 Quick Start

We provide pre-trained base policies and CFN checkpoints on 🤗 Hugging Face, allowing you to directly evaluate TACO without training.

### Download Pre-trained Models

```bash
# Download base policy checkpoints & CFN checkpoints
### CFN state dict is saved at `cfns` sub-directory in the repo
hf download rhodes-team-teleai/pi05_TACO_libero_finetuned --local-dir /path/to/your/dir --max-workers 16
```

## 📊 Usage

### Pi0.5
#### Eval in Robotwin

This branch runs RoboTwin through a pinned official checkout and a
manifest-driven pipeline rather than the upstream single-task example scripts,
so the steps below differ from the rest of this README. Start with
[ROBOTWIN12.md](./ROBOTWIN12.md).

The upstream example flow (`task_dataset_collection.sh` and
`rt2-hdf5_2_hdf5_2_lerobot.sh`) reached into a vendored RoboTwin under
`third_party/Robotwin` and was removed with it: it drives one hardcoded task
(`adjust_bottle`) and cannot run the multi-task pipeline, which needs the
official checkout's `data/decode_image_bit.py` and
`scripts/eval_policy_xpolicylab.py`. To collect and convert a task here:

```
python scripts/robotwin_multitask.py collect-convert --task click_bell
```

`data_trans/v21_to_v30.sh` and `data_trans/make_sure_stats.sh` remain, but the
converter writes LeRobot v3 directly, so neither is part of this path.

You can fine-tune your own pi0.5:
```
bash ./third_party/lerobot/scripts/train_pi05.sh
```

or just use our trained pi0.5 model on Huggingface.

2. Collect internal representation

Modify and run:
```
bash ./scripts/collect_inernal_representation/pi05_robotwin2/collect.sh
```

3. Train CFN

Modify and run:
```
bash ./scripts/train_cfn/train_cfn_example.sh
```

4. Eval TACO

```
python scripts/robotwin_multitask.py --policy-server 127.0.0.1:18080 queue
python scripts/robotwin_multitask.py summarize
```

The upstream single-task entry point (`eval_robotwin2_torch_pi05_taco.sh`) was
removed with the vendored RoboTwin; see [ROBOTWIN12.md](./ROBOTWIN12.md).

#### Eval in Libero
1. Collect internal representation

Run:
```
bash ./scripts/collect_inernal_representation/pi05_libero/collect.sh
```

2. Train CFN

Modify and run:
```
bash ./scripts/train_cfn/train_cfn_example.sh
```

3. Eval TACO

Modify and run:
```
bash ./scripts/eval/eval_libero_pi05_taco.sh
```

### Openvla
#### Eval in Libero
1. Collect internal representation

Download `openvla/modified_libero_rlds` from Huggingface.

Modify and run:
```
bash ./scripts/collect_inernal_representation/openvla_libero/collect.sh
```

2. Train CFN

Modify and run:
```
bash ./scripts/train_cfn/train_cfn_example.sh
```

3. Eval in Libero

Modify and run:
```
bash ./scripts/eval/libero_openvla_taco/eval_libero_openvla_taco.sh
```

## 📈 Results

### Performance on Simpler-WindowX Benchmark

Evaluation of success rates (%) on Simpler-WindowX tasks. We compare **$\pi_0$ + TACO** against the base policy and other state-of-the-art methods.

| Method | Spoon on Towel | Carrot on Plate | Stack Cubes | Eggplant in Basket | Average |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RT-1-X | 0.0% | 4.2% | 0.0% | 0.0% | 1.1% |
| Octo | 12.5% | 8.3% | 0.0% | 43.1% | 16.0% |
| RoboVLM | 29.2% | 25.0% | 12.5% | 58.3% | 31.3% |
| SpatialVLA | 16.7% | 25.0% | 29.2% | **100.0%** | 42.7% |
| $\pi_0$ (Base) | 36.0% | 42.0% | **34.0%** | 80.0% | 48.0% |
| **$\pi_0$ + TACO (Ours)** | **52.0%** | **52.0%** | 30.0% | 88.0% | **55.5%** |

### Performance on Libero-Long Benchmark

Our method consistently improves performance across both Flow-Matching ($\pi_{0.5}$) and Autoregressive (OpenVLA) architectures on long-horizon tasks.

| Task | $\pi_{0.5}$ (Base) | **$\pi_{0.5}$ + TACO** | OpenVLA (Base) | **OpenVLA + TACO** |
| :--- | :---: | :---: | :---: | :---: |
| Soup and Sauce in Basket | 98.0% | **100.0%** | 60.0% | **66.0%** |
| Cheese and Butter in Basket | **100.0%** | 96.0% | 76.0% | **82.0%** |
| Turn on Stove and Place Moka | 98.0% | 98.0% | 58.0% | 52.0% |
| Black Bowl in Drawer | 98.0% | **100.0%** | 36.0% | **50.0%** |
| Mugs on Plates | 98.0% | 98.0% | 32.0% | **50.0%** |
| Book in Caddy | 100.0% | 100.0% | 82.0% | **90.0%** |
| Mug and Pudding on Plate | 96.0% | 92.0% | **60.0%** | 54.0% |
| Soup and Cheese in Basket | 94.0% | **100.0%** | 70.0% | **80.0%** |
| Moka Pots on Stove | 68.0% | **86.0%** | 20.0% | **28.0%** |
| Mug in Microwave | 98.0% | 96.0% | 46.0% | **48.0%** |
| **Average** | 94.8% | **96.6%** | 54.0% | **60.0%** |

### Performance on RoboTwin 1.0 Benchmark

| Task | $\pi_0$ (Base) | **$\pi_0$ + TACO** | Improvement |
| :--- | :---: | :---: | :---: |
| Block Handover | 41.0% | **62.0%** | +21.0% |
| Bottles Adjust | 31.0% | **40.0%** | +9.0% |
| Container Place | 25.0% | **40.0%** | +15.0% |
| Diverse Bottles Pick | 21.0% | **27.0%** | +6.0% |
| Dual Bottles Pick Easy | 60.0% | **70.0%** | +10.0% |
| Dual Bottles Pick Hard | 48.0% | **52.0%** | +4.0% |
| Pick Apple Messy | 15.0% | **19.0%** | +4.0% |
| Shoe Place | 42.0% | **50.0%** | +8.0% |
| Mug Hanging Easy | 7.0% | **12.0%** | +5.0% |
| **Average** | 32.2% | **41.3%** | **+9.1%** |

### Performance on RoboTwin 2.0 Benchmark

Evaluation on the **RoboTwin 2.0** benchmark. We report the success rate improvement of TACO over the $\pi_{0.5}$ baseline. The table highlights the top 5 tasks with the most significant gains.

| Task | $\pi_{0.5}$ (Base) | **$\pi_{0.5}$ + TACO** | Improvement |
| :--- | :---: | :---: | :---: |
| Move Can Pot | 42.0% | **57.0%** | +15.0% |
| Handover Block | 24.0% | **36.0%** | +12.0% |
| Place Shoe | 53.0% | **65.0%** | +12.0% |
| Stamp Seal | 26.0% | **38.0%** | +12.0% |
| Beat Block Hammer | 69.0% | **79.0%** | +10.0% |
| ... | ... | ... | ... |
| **Average (41 tasks)** | 59.3% | **64.0%** | +4.7% |

## 📁 Project Structure

```
TACO/
├── cfn/                         # Coin Flipping implementation
│   ├── cfn_net.py               # CFN model architecture
│   └── feature_dataset.py       # Dataset for training CFN
├── scripts/
│   ├── collect_internal_representation/  # Scripts to collect features
│   ├── train_cfn/               # CFN training scripts
│   └── eval/                    # Evaluation scripts
├── third_party/
│   ├── lerobot/                 # Pi0.5 (LeRobot) implementation & LeRobot-Libero evaluation
│   ├── openvla/                 # OpenVLA implementation
│   └── Robotwin/                # Robotwin environment
└── README.md
```

## 🎓 Citation

If you find TACO useful for your research, please cite our paper:

```bibtex
@article{yang2025taco,
  title={Steering Vision-Language-Action Models as Anti-Exploration: A Test-Time Scaling Approach},
  author={Siyuan Yang, Yang Zhang, Haoran He, Ling Pan, Xiu Li, Chenjia Bai, Xuelong Li},
  journal={arXiv preprint arXiv:2512.02834},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) for the Pi0.5 implementation
- [OpenVLA](https://github.com/openvla/openvla) for the OpenVLA base model
- [Robotwin](https://robotwin-platform.github.io/) for the simulation environment
- [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO) for the benchmark tasks

## 📧 Contact

For questions or collaborations, please contact:
- Email: breezeyoung9470@gmail.com
- GitHub Issues: [Open an issue](https://github.com/breez3young/TACO/issues)
