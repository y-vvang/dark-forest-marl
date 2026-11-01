#!/usr/bin/env python3
"""Config + env factory for DarkForestGrid v2.0."""
from __future__ import annotations

import copy
import os

import numpy as np
import yaml

import env2_v25 as env_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "config_v25.yaml")
_OBS_DIM = env_mod.OBS_DIM
_N_ACTIONS = env_mod.N_ACTIONS
OBS_DIM = _OBS_DIM
N_ACTIONS = _N_ACTIONS
ACTION_NAMES = env_mod.ACTION_NAMES


def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def apply_ablation(cfg: dict, ablation: str) -> dict:
    cfg = copy.deepcopy(cfg)
    if ablation in ("main", "mixed"):
        return cfg
    ab = cfg.get("ablations", {}).get(ablation)
    if ab is None:
        raise ValueError(f"unknown ablation: {ablation}")
    cfg["env"].update(ab)
    return cfg


def make_env(cfg: dict, seed: int = 0) -> "env_mod.DarkForestEnv2":
    e = cfg["env"]
    return env_mod.DarkForestEnv2(
        n_agents=int(e["n_agents"]), world=float(e["world"]),
        n_norm=float(e.get("n_norm", 36.0)),
        detect_radius=float(e["detect_radius"]), t_max=int(e["t_max"]),
        p_detect_base=float(e["p_detect_base"]), stealth_mult=float(e["stealth_mult"]),
        p_detect_max=float(e["p_detect_max"]),
        p_explosion=float(e["p_explosion"]), develop_rate=float(e["develop_rate"]),
        tech_cap=float(e["tech_cap"]),
        tech_spawn_lo=float(e["tech_spawn_lo"]), tech_spawn_hi=float(e["tech_spawn_hi"]),
        kill_base=float(e["kill_base"]), kill_tech_slope=float(e["kill_tech_slope"]),
        res_min=float(e["res_min"]), res_max=float(e["res_max"]),
        p_deplete=float(e["p_deplete"]), deplete_lo=float(e["deplete_lo"]), deplete_hi=float(e["deplete_hi"]),
        gate_broadcast_tech=float(e["gate_broadcast_tech"]), gate_broadcast_res=float(e["gate_broadcast_res"]),
        gate_observe_tech=float(e["gate_observe_tech"]), gate_domain_tech=float(e["gate_domain_tech"]),
        gate_strike_tech=float(e["gate_strike_tech"]),
        develop_reward=float(e["develop_reward"]), death_penalty=float(e["death_penalty"]),
        bd_entry_penalty=float(e["bd_entry_penalty"]),
        bd_growth_mult=float(e["bd_growth_mult"]), bd_explosion_mult=float(e["bd_explosion_mult"]),
        enable_black_domain=bool(e["enable_black_domain"]),
        spawn_schedule=[(int(t), int(c)) for t, c in e.get("spawn_schedule", [[0, 12]])],
        res_total_budget=float(e.get("res_total_budget", 150.0)),
        shield_cost_frac=float(e.get("shield_cost_frac", 0.5)),
        softsign_a=float(e.get("softsign_a", 0.2)),
        softsign_reward_scale=float(e.get("softsign_reward_scale", 1.0)),
        res_reward_scale=float(e.get("res_reward_scale", 0.0)),
        attack_reward=float(e.get("attack_reward", 0.0)),
        seed=seed,
    )


# spawn-tech classes for stratified analysis (point 7 of the v2.0 redesign)
def tech_class(tech: float) -> str:
    if tech < 3.5:
        return "low"
    if tech < 7.0:
        return "mid"
    return "high"
