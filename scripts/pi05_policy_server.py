#!/usr/bin/env python3
"""Serve one PI0.5 instance to concurrent RoboTwin env and feature clients."""

from __future__ import annotations

import argparse
import socketserver
import threading
import traceback
from pathlib import Path

import numpy as np
import torch

from cfn.cfn_net import CFN
from cfn.pi05_rpc import receive_message, send_message
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


# TACO scores 50 noise candidates per step. Running all 50 through the model at
# once leaves the caching allocator holding the whole peak for the lifetime of
# the process — measured at ~21 GiB resident for a 3B model that needs ~7 GiB.
# Scoring in slices keeps the same candidates, in the same order, at a fraction
# of the peak.
NOISE_CANDIDATES = 50
CANDIDATES_PER_SLICE = 10
# The pool a request gets when it does not carry `noise_seed`. Keeping the
# default at 42 pins every result produced before seed selection existed.
DEFAULT_NOISE_SEED = 42


class PI05Core:
    def __init__(self, policy_path: Path, tokenizer_path: Path, cfn_checkpoint: Path | None):
        self.policy = PI05Policy.from_pretrained(policy_path, local_files_only=True).eval()
        config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=policy_path,
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)}
            },
        )
        parameter = next(self.policy.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.lock = threading.Lock()
        self.cfn = None
        self.noise_shape = (
            NOISE_CANDIDATES,
            self.policy.model.config.n_action_steps,
            self.policy.model.config.max_action_dim,
        )
        self.noise_pools = {}
        self.noise = self._noise_pool(DEFAULT_NOISE_SEED)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.initial_cpu_rng_state = torch.get_rng_state()
        self.initial_cuda_rng_state = torch.cuda.get_rng_state(self.device)
        self.session_rng_states = {}
        if cfn_checkpoint is not None:
            self.load_cfn(cfn_checkpoint)

    def _noise_pool(self, seed: int):
        """Candidate pool for one seed, built once and cached (≈320 KB each).

        Drawn from an explicit generator rather than the global RNG, so a pool's
        contents depend only on (seed, shape) — not on process history or on
        when it was first requested. That is what lets a re-run with a new seed
        be a genuine replicate of the same algorithm.
        """
        pool = self.noise_pools.get(seed)
        if pool is None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            pool = torch.randn(self.noise_shape, generator=generator, dtype=torch.float32).to(
                self.device, dtype=self.dtype
            )
            self.noise_pools[seed] = pool
        return pool

    def load_cfn(self, checkpoint: Path):
        cfn = CFN(cfn_output_dim=20, cfn_hidden_dim=1536).to(self.device)
        cfn.cfn.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.cfn = cfn.cfn.eval()
        return {"checkpoint": str(checkpoint)}

    def _to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _env_batch(self, request: dict, repeats: int) -> dict:
        front, right, left = request["images"]

        def image_tensor(image):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image, got {array.shape}")
            return torch.from_numpy(np.transpose(array, (2, 0, 1)) / 255.0).to(self.dtype)

        batch = {
            "observation.state": torch.from_numpy(np.asarray(request["state"])).to(self.dtype).unsqueeze(0),
            "observation.images.cam_high": image_tensor(front).unsqueeze(0),
            "observation.images.cam_left_wrist": image_tensor(left).unsqueeze(0),
            "observation.images.cam_right_wrist": image_tensor(right).unsqueeze(0),
            "task": [request["instruction"]],
        }
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.repeat(repeats, *([1] * (value.ndim - 1))).to(self.device)
            else:
                batch[key] = value * repeats
        return batch

    def action(self, request: dict):
        mode = request["mode"]
        repeats = 1 if mode == "baseline" else NOISE_CANDIDATES
        with self.lock, torch.inference_mode():
            if mode == "baseline":
                batch = self._to_device(self.preprocessor(self._env_batch(request, repeats)))
                session_id = request["session_id"]
                cpu_state, cuda_state = self.session_rng_states.get(
                    session_id, (self.initial_cpu_rng_state, self.initial_cuda_rng_state)
                )
                torch.set_rng_state(cpu_state)
                torch.cuda.set_rng_state(cuda_state, self.device)
                actions = self.policy.predict_action_chunk(batch)
                self.session_rng_states[session_id] = (
                    torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
                )
                selected = self.postprocessor(actions)[0]
            elif mode == "taco":
                if self.cfn is None:
                    raise RuntimeError("TACO action requested before a CFN checkpoint was loaded")
                cfn_dtype = next(self.cfn.parameters()).dtype
                pool = self._noise_pool(int(request.get("noise_seed", DEFAULT_NOISE_SEED)))
                candidates = []
                scores = []
                for start in range(0, repeats, CANDIDATES_PER_SLICE):
                    stop = min(start + CANDIDATES_PER_SLICE, repeats)
                    slice_batch = self._to_device(self.preprocessor(self._env_batch(request, stop - start)))
                    slice_actions, slice_features = self.policy.predict_action_chunk_and_get_feature(
                        slice_batch, pool[start:stop].clone()
                    )
                    # The postprocessor's relative-to-absolute step reuses the
                    # observation state cached by the last preprocessor call, so
                    # a slice must be converted in the slice it was inferred in.
                    candidates.append(self.postprocessor(slice_actions))
                    scores.append(self.cfn(slice_features.to(cfn_dtype)).norm(dim=1))
                selected_index = int(torch.argmin(torch.cat(scores)).item())
                selected = torch.cat(candidates, dim=0)[selected_index]
            else:
                raise ValueError(f"Unknown inference mode: {mode}")
        return selected.float().cpu().numpy()

    def feature(self, request: dict):
        def tensor(value):
            return torch.from_numpy(np.asarray(value)).to(self.dtype).unsqueeze(0)

        batch = {
            "observation.state": tensor(request["state"]),
            "observation.images.cam_high": tensor(request["images"][0]),
            "observation.images.cam_left_wrist": tensor(request["images"][1]),
            "observation.images.cam_right_wrist": tensor(request["images"][2]),
            "action": tensor(request["action"]),
            "task": [request["instruction"]],
        }
        def repeat(count: int) -> dict:
            widened = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    widened[key] = value.repeat(count, *([1] * (value.ndim - 1))).to(self.device)
                else:
                    widened[key] = value * count
            return widened

        with self.lock, torch.inference_mode():
            distances = []
            features_scored = []
            for start in range(0, NOISE_CANDIDATES, CANDIDATES_PER_SLICE):
                stop = min(start + CANDIDATES_PER_SLICE, NOISE_CANDIDATES)
                slice_batch = self._to_device(self.preprocessor(repeat(stop - start)))
                actions, features = self.policy.predict_action_chunk_and_get_feature(
                    slice_batch, self.noise[start:stop].clone()
                )
                distances.append(torch.norm(actions - slice_batch["action"], dim=(1, 2), p=2))
                features_scored.append(features)
            selected = torch.cat(features_scored)[torch.argmin(torch.cat(distances))]
        return selected.float().cpu().numpy()

    def dispatch(self, request: dict):
        operation = request["operation"]
        if operation == "health":
            return {
                "device": str(self.device),
                "gpu": torch.cuda.get_device_name(self.device),
                "cfn_loaded": self.cfn is not None,
            }
        if operation == "load_cfn":
            with self.lock:
                return self.load_cfn(Path(request["checkpoint"]))
        if operation == "action":
            return self.action(request)
        if operation == "feature":
            return self.feature(request)
        raise ValueError(f"Unknown operation: {operation}")


class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                request = receive_message(self.request)
            except ConnectionError:
                return
            try:
                result = self.server.core.dispatch(request)
                send_message(self.request, {"ok": True, "result": result})
            except Exception as exc:
                send_message(self.request, {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                })


class ThreadingPolicyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, core):
        self.core = core
        super().__init__(address, RequestHandler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--cfn-checkpoint", type=Path)
    args = parser.parse_args()
    core = PI05Core(args.policy_path, args.tokenizer_path, args.cfn_checkpoint)
    server = ThreadingPolicyServer((args.host, args.port), core)
    print({"address": f"{args.host}:{args.port}", **core.dispatch({"operation": "health"})}, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
