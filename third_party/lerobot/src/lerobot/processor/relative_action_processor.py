"""Relative/absolute action processors used by converted PI0.5 checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_STATE

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if state.ndim == 3:
        state = state[:, 0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    result = actions.clone()
    result[..., :dims] -= state_offset
    return result


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if state.ndim == 3:
        state = state[:, 0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    result = actions.clone()
    result[..., :dims] += state_offset
    return result


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim
        tokens = [str(name).lower() for name in self.exclude_joints if name]
        mask = [
            not any(token == str(name).lower() or token in str(name).lower() for token in tokens)
            for name in self.action_names[:action_dim]
        ]
        mask.extend([True] * (action_dim - len(mask)))
        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state
        if not self.enabled:
            return transition
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is not None and state is not None:
            result[TransitionKey.ACTION] = to_relative_actions(
                action, state, self._build_mask(action.shape[-1])
            )
        return result

    def reset(self) -> None:
        self._last_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]):
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None or self.relative_step._last_state is None:
            raise RuntimeError("Absolute action processor has no cached observation state")
        result = transition.copy()
        action = result.get(TransitionKey.ACTION)
        if action is not None:
            result[TransitionKey.ACTION] = to_absolute_actions(
                action,
                self.relative_step._last_state,
                self.relative_step._build_mask(action.shape[-1]),
            )
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]):
        return features
