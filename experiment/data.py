"""Environment factory — the single binding point between config.yaml and
DarkForestGrid v1.1.

Per data_contract.md: the "dataset" of this run IS the self-built simulation
environment. No external data file is read anywhere in the pipeline; everything
(seed control, ablation overrides, obs/action constants) is bound here so the run
stays auditable. Ablation overrides have ONE source of truth: config.yaml.
"""
from __future__ import annotations

import os

import yaml

from env import ACTION_NAMES, N_ACTIONS, OBS_DIM, DarkForestEnv  # noqa: F401  (re-exported)


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def ablation_names(cfg: dict) -> list[str]:
    return list(cfg.get("ablations", {}).keys())


def make_env(cfg: dict, seed: int, ablation: str | None = None) -> DarkForestEnv:
    """Build DarkForestGrid v1.1 with config values; apply ablation overrides."""
    env_cfg = dict(cfg["env"])
    if ablation is not None:
        if ablation == "main":
            pass
        elif ablation in cfg.get("ablations", {}):
            env_cfg.update(cfg["ablations"][ablation])
        else:
            raise ValueError(f"unknown ablation: {ablation}")
    return DarkForestEnv(seed=seed, **env_cfg)
