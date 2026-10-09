#!/usr/bin/env python3
"""DarkForestGrid v2.0 — resource-based existential pressure, invasion economy,
tech-gated action pool, and two-phase decision resolution.

Redesign rationale (user-co-designed, 2026-09-07) — replaces v1.x mechanisms:
  1. DEATH PENALTY replaces survival reward: dying costs -10, an order of magnitude
     heavier than any other reward/penalty (develop +0.05, domain entry -0.5).
  2. RESOURCES replace territorial crowding as the existential channel:
     - each civilization spawns with resources ~ U[res_min, res_max]
     - each tick, with small probability a civilization loses a random amount
     - resources hitting 0 kills the civilization (silent, exogenous)
     - resources are NOT replenishable except by INVASION (loot = half the victim's
       resources). Resources carry no direct reward/penalty — they buy survival time.
     - INVADE (no tech gate): guaranteed kill of a known, non-shielded target — but
       the target FIRST discovers the invader and completes one action with that
       knowledge (re-choose under invasion). If the dying target invades back,
       BOTH die (mutual destruction). No tech loss on strikes (v1.x costs removed).
  3. OBSERVE (tech > 5, immediate): reveals the target's exact tech and resources;
     the v1.4 "discovery exposes you" mechanism moved here — after observing, a
     random other civilization detects you next tick. Passive detection no longer
     exposes the detector.
  4. BROADCAST is target-directed (second decision pass) and DELAYED one tick
     ("this turn it is initiated, next turn the target is exposed") — a broadcast
     declared by a dying civilization still resolves after its death (dying curse).
  5. TECH-GATED ACTION POOL (primitive civilizations cannot use advanced actions):
       BLACK_DOMAIN  tech > 5   declared this tick, ACTIVE NEXT tick (windup)
       BROADCAST_*   tech > 1 or resources > broadcast_res_gate; resolves next tick
       STRIKE        tech > 7   immediate
       OBSERVE       tech > 5   immediate (exposure next tick)
     Tech spawns uniformly over the full gate-relevant range U[0.5, 10] (cap 10),
     so spawn class (primitive / mid / advanced) is a strategic identity. Analysis
     must stratify by spawn tech (see evaluate.py).
  6. Strike tech costs removed (no attacker cost, no failed-strike damage).
  7. (see 5) — spawn tech random over the full range; stratified analysis required.
  8. Branch experiment FIRST: heterogeneous training — random / DF-heuristic /
     pacifist agents mixed into training alongside learning agents (train2.py
     --mode mixed).

Black domain (carried from v1.4, user spec): irreversible, absolute sanctuary —
undetectable, blind, immune to strikes AND invasions; broadcasting/striking/
invading/observing forbidden inside; entry costs -0.5 (instantaneous); growth x0.2
and explosion prob x0.1 inside; resource depletion still applies (camping is a
countdown, not a strategy). Declaring costs one exposed windup tick.

Action space (8): SILENT, DEVELOP, BROADCAST_SELF, BROADCAST_OTHER*, STRIKE,
BLACK_DOMAIN, INVADE*, OBSERVE   (* second-pass target selection among known alive,
non-shielded civilizations). STRIKE/OBSERVE auto-target the nearest known alive.

Multi-phase tick protocol (see step()): 
  1. `select_targets(actions)` -> candidate masks for INVADE / BROADCAST_OTHER
  2. caller provides targets; `begin_tick(actions, targets)` -> invasion notices
  3. caller re-chooses actions for invaded civilizations (they now know the invader)
  4. `resolve(re_actions)` -> full resolution (kills, loot, resources, growth...)
"""
from __future__ import annotations

import numpy as np

ACTION_NAMES = ["SILENT", "DEVELOP", "BROADCAST_SELF", "BROADCAST_OTHER(retired)", "STRIKE",
                "BLACK_DOMAIN", "INVADE", "OBSERVE", "STRIKE_WEAK", "STRIKE_STRONG"]
(A_SILENT, A_DEVELOP, A_BSELF, A_BOTHER, A_STRIKE, A_BDOMAIN, A_INVADE, A_OBSERVE) = range(8)
# v2.3: strikes gain target choice — three coordinate-selection modes over KNOWN
# civilizations (nearest / weakest / strongest). BROADCAST_OTHER is retired:
# its index stays reserved but is invalid in every state (broadcasts are now
# all-broadcast only, via BROADCAST_SELF).
(A_STRIKE_WEAK, A_STRIKE_STRONG) = (8, 9)
N_ACTIONS = 10

NEAR_K = 3
OBS_DIM = 7 + N_ACTIONS + NEAR_K * 7      # 15 + 21 = 36


class DarkForestEnv2:
    """Vectorised multi-agent Dark Forest environment, v2.0 (numpy)."""

    def __init__(
        self,
        n_agents: int = 12,
        world: float = 100.0,
        detect_radius: float = 30.0,
        t_max: int = 250,
        p_detect_base: float = 0.03,
        stealth_mult: float = 0.25,
        p_detect_max: float = 0.35,
        p_explosion: float = 0.01,
        develop_rate: float = 0.05,
        tech_cap: float = 10.0,
        tech_spawn_lo: float = 0.5,
        tech_spawn_hi: float = 10.0,
        kill_base: float = 0.55,
        kill_tech_slope: float = 0.12,
        res_min: float = 5.0,
        res_max: float = 20.0,
        p_deplete: float = 0.02,
        deplete_lo: float = 0.5,
        deplete_hi: float = 2.0,
        gate_broadcast_tech: float = 1.0,
        gate_broadcast_res: float = 10.0,
        gate_observe_tech: float = 5.0,
        gate_domain_tech: float = 5.0,
        gate_strike_tech: float = 7.0,
        develop_reward: float = 0.05,
        death_penalty: float = 10.0,
        bd_entry_penalty: float = 0.5,
        bd_growth_mult: float = 0.2,
        bd_explosion_mult: float = 0.1,
        enable_black_domain: bool = True,
        spawn_schedule: list | None = None,
        res_total_budget: float = 150.0,
        shield_cost_frac: float = 0.5,
        softsign_a: float = 0.2,
        softsign_reward_scale: float = 1.0,
        seed: int = 0,
    ):
        self.n = n_agents
        self.world = float(world)
        self.detect_radius = float(detect_radius)
        self.t_max = int(t_max)
        self.p_detect_base = float(p_detect_base)
        self.stealth_mult = float(stealth_mult)
        self.p_detect_max = float(p_detect_max)
        self.p_explosion = float(p_explosion)
        self.develop_rate = float(develop_rate)
        self.tech_cap = float(tech_cap)
        self.tech_spawn_lo = float(tech_spawn_lo)
        self.tech_spawn_hi = float(tech_spawn_hi)
        self.kill_base = float(kill_base)
        self.kill_tech_slope = float(kill_tech_slope)
        self.res_min = float(res_min)
        self.res_max = float(res_max)
        self.p_deplete = float(p_deplete)
        self.deplete_lo = float(deplete_lo)
        self.deplete_hi = float(deplete_hi)
        self.gate_broadcast_tech = float(gate_broadcast_tech)
        self.gate_broadcast_res = float(gate_broadcast_res)
        self.gate_observe_tech = float(gate_observe_tech)
        self.gate_domain_tech = float(gate_domain_tech)
        self.gate_strike_tech = float(gate_strike_tech)
        self.develop_reward = float(develop_reward)
        self.death_penalty = float(death_penalty)
        self.bd_entry_penalty = float(bd_entry_penalty)
        self.bd_growth_mult = float(bd_growth_mult)
        self.bd_explosion_mult = float(bd_explosion_mult)
        self.enable_black_domain = bool(enable_black_domain)
        self.spawn_schedule = [(int(t), int(c)) for t, c in (spawn_schedule or [(0, 12)])]
        self.res_total_budget = float(res_total_budget)
        self.shield_cost_frac = float(shield_cost_frac)
        self.softsign_a = float(softsign_a)
        self.softsign_reward_scale = float(softsign_reward_scale)
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        # transient tick state
        self._invaders: dict[int, int] = {}       # victim -> original invader
        self._targets: dict[int, int] = {}        # actor -> chosen target
        self._actions: np.ndarray | None = None

    # ------------------------------------------------------------------ utils
    def _pair_dist(self) -> np.ndarray:
        delta = np.abs(self.pos[:, None, :] - self.pos[None, :, :])
        delta = np.minimum(delta, self.world - delta)
        return np.linalg.norm(delta, axis=-1)

    def reset(self, seed: int | None = None):
        n = self.n
        rng = self.rng if seed is None else np.random.default_rng(seed)
        if seed is not None:
            self.rng = rng
        pos = rng.uniform(0, self.world, size=(n, 2))
        for _ in range(400):
            d = self._pair_dist() if hasattr(self, "pos") else np.full((n, n), np.inf)
            bad = ~np.eye(n, dtype=bool) & (d < 12.0)
            if not bad.any():
                break
            for i, j in zip(*np.where(np.triu(bad))):
                pos[i] = rng.uniform(0, self.world, size=2)
                break
        self.pos = pos
        # v2.2 resource conservation (Axiom 2, user-co-designed): the world has a
        # FIXED total resource budget. Any matter leaving circulation (depletion,
        # unlooted corpses, shield costs) accrues to a FREE POOL. New civilizations
        # are born ENDENOUSLY: after each birth the next civ's spawn resource is
        # pre-rolled; it is born the moment the pool can afford it (pool -= spawn_res).
        self.res_total = float(self.res_total_budget)
        self.spawn_tick = np.zeros(n, dtype=np.int64)
        n0 = self.spawn_schedule[0][1] if self.spawn_schedule else n
        self.spawn_tick[n0:] = 10**9          # not yet born (pool-gated queue)
        self.tech = np.zeros(n); self.res = np.zeros(n)
        tech0 = rng.uniform(self.tech_spawn_lo, self.tech_spawn_hi, size=n0)
        res0 = rng.uniform(self.res_min, self.res_max, size=n0)
        if res0.sum() > self.res_total:
            res0 = res0 * (self.res_total / res0.sum())   # initial cohort fits the budget
        self.tech[:n0], self.res[:n0] = tech0, res0
        self.free_pool = self.res_total - float(self.res.sum())
        # v2.3 unbounded births: exactly ONE next spawn-resource amount is
        # pre-rolled; each birth pre-rolls the next — the queue never empties
        self.pending_spawns: list[float] = [
            float(rng.uniform(self.res_min, self.res_max))]
        self.spawn_tech = self.tech.copy()
        self.spawn_res = self.res.copy()
        self.birth_log: list[tuple[int, int, float]] = []   # (tick, slot, spawn_res)
        self.death_cause = [""] * n          # "" alive/unborn; strike|invade|mutual|starve
        self.peak_tech = self.tech.copy()    # max tech reached while alive (evidence)
        self.alive = np.zeros(n, dtype=bool)
        self.alive[:n0] = True                     # later cohorts: born on schedule
        self.known = np.zeros((n, n), dtype=bool)
        self.observed = np.zeros((n, n), dtype=bool)     # intel: exact tech/res of j
        self.public = np.zeros(n, dtype=bool)
        self.in_domain = np.zeros(n, dtype=bool)
        self.domain_windup = np.zeros(n, dtype=bool)     # declared, active next tick
        self.pending_reveal = []                          # (revealer_kind, target_j) resolve next tick
        self.pending_counter: list[int] = []              # observers awaiting exposure
        self.last_action = np.full(n, A_SILENT, dtype=np.int64)
        self.first_detect_tick = -np.ones((n, n), dtype=np.int64)
        self.t = 0
        return self._obs(), self._valid()

    # ------------------------------------------------------- staggered spawns
    @property
    def exists(self) -> np.ndarray:
        """A civilization slot is ACTIVE only after its spawn tick."""
        return self.alive & (self.t >= self.spawn_tick)

    def _activate_pending_spawns(self):
        """v2.3: UNBOUNDED endogenous births. Exactly one spawn-resource amount is
        pre-rolled at any time; when a birth fires, the NEXT amount is pre-rolled —
        the queue never empties, so civilization count has no slot-based cap beyond
        the conservation budget. One birth per tick max; the pool is debited."""
        if not self.pending_spawns:
            return
        if self.free_pool < self.pending_spawns[0]:
            return
        slots = np.where(self.spawn_tick > 10**8)[0]
        if slots.size == 0:
            return
        slot = int(slots[0])
        spawn_res = self.pending_spawns.pop(0)
        self.pending_spawns.append(float(self.rng.uniform(self.res_min, self.res_max)))
        act_tech = self.tech[self.alive]
        if act_tech.size >= 2:
            lo, hi = np.percentile(act_tech, [25, 75])
        else:
            lo, hi = self.tech_spawn_lo, self.tech_spawn_hi
        if hi - lo < 1.0:                    # minimal spread for a viable cohort
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5, mid + 0.5
        self.tech[slot] = self.rng.uniform(lo, hi)
        self.res[slot] = spawn_res
        self.free_pool -= spawn_res
        self.spawn_tech[slot] = self.tech[slot]
        self.spawn_res[slot] = spawn_res
        self.spawn_tick[slot] = self.t + 1
        self.alive[slot] = True
        self.birth_log.append((self.t + 1, slot, spawn_res))

    # ------------------------------------------------------------------ masks
    def _valid(self) -> np.ndarray:
        n = self.n
        strikeable = (self.known & self.alive[None, :] & (~self.in_domain)[None, :]
                      & (~self.domain_windup)[None, :]).any(axis=1) & self.alive
        valid = np.ones((n, N_ACTIONS), dtype=bool)
        valid[:, A_BOTHER] = False            # v2.3: targeted broadcast retired
        valid[:, A_STRIKE] = strikeable
        valid[:, A_STRIKE_WEAK] = strikeable
        valid[:, A_STRIKE_STRONG] = strikeable
        valid[:, A_INVADE] = strikeable
        valid[:, A_OBSERVE] = strikeable
        valid[:, A_BSELF] &= (self.tech > self.gate_broadcast_tech) | (self.res > self.gate_broadcast_res)
        valid[:, A_BOTHER] &= (self.tech > self.gate_broadcast_tech) | (self.res > self.gate_broadcast_res)
        valid[:, A_OBSERVE] &= self.tech > self.gate_observe_tech
        valid[:, A_BDOMAIN] = self.enable_black_domain & (self.tech > self.gate_domain_tech) & self.alive
        valid[:, A_STRIKE] &= self.tech > self.gate_strike_tech
        valid[:, A_STRIKE_WEAK] &= self.tech > self.gate_strike_tech
        valid[:, A_STRIKE_STRONG] &= self.tech > self.gate_strike_tech
        shielded = self.in_domain | self.domain_windup
        valid[shielded, A_BOTHER] = False
        valid[shielded, A_STRIKE] = False
        valid[shielded, A_STRIKE_WEAK] = False
        valid[shielded, A_STRIKE_STRONG] = False
        valid[shielded, A_INVADE] = False
        valid[shielded, A_OBSERVE] = False
        valid[shielded, A_BSELF] = False
        # unborn civilizations take no actions this tick
        valid[~self.alive, :] = False
        valid[~self.alive, A_SILENT] = True
        return valid

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        n = self.n
        dist = self._pair_dist()
        crowd = ((dist <= self.detect_radius) & self.alive[None, :] & ~np.eye(n, dtype=bool)).sum(axis=1)
        onehot = np.zeros((n, N_ACTIONS), dtype=np.float64)
        onehot[np.arange(n), self.last_action] = 1.0

        # v2.2: tech is unbounded — observe it through log1p (reference scale 50)
        LOG_REF = np.log1p(50.0)
        scal = np.concatenate([
            (np.log1p(np.maximum(self.tech, 0.0)) / LOG_REF)[:, None],
            (np.maximum(self.res, 0.0) / self.res_max)[:, None],
            ((self.known & self.alive[None, :]).sum(1) / (n - 1))[:, None],
            np.full((n, 1), self.alive.sum() / n),
            np.full((n, 1), self.t / self.t_max),
            self.in_domain.astype(np.float64)[:, None],
            self.domain_windup.astype(np.float64)[:, None],
        ], axis=1)

        big = np.where(dist <= self.detect_radius, dist, np.inf)
        np.fill_diagonal(big, np.inf)
        big = np.where(self.known & self.alive[None, :], big, np.inf)
        order = np.argsort(big, axis=1)[:, :NEAR_K]
        idx = np.arange(n)[:, None]
        known_pair = np.take_along_axis(self.known & self.alive[None, :] & (~self.in_domain)[None, :], order, axis=1)
        delta = self.pos[order] - self.pos[idx]
        delta -= self.world * np.round(delta / self.world)
        dmask = np.isfinite(np.take_along_axis(big, order, axis=1))
        near_tech = np.where(known_pair & self.observed[idx, order], self.tech[order], -1.0)
        near_res = np.where(known_pair & self.observed[idx, order], self.res[order], -1.0)
        near_d = np.where(dmask, np.take_along_axis(big, order, axis=1), 0.0) / 50.0
        near = np.concatenate([
            (delta[..., 0] / 50.0) * dmask,
            (delta[..., 1] / 50.0) * dmask,
            near_d,
            np.where(near_tech >= 0, np.log1p(np.maximum(near_tech, 0.0)) / LOG_REF, near_tech),
            (near_res / self.res_max),
            self.public[order].astype(np.float64) * known_pair,
            self.in_domain[order].astype(np.float64) * known_pair,
        ], axis=-1)
        obs = np.concatenate([scal, onehot, near], axis=1).astype(np.float32)
        obs[~self.alive] = 0.0
        return obs

    # ----------------------------------------------------- two-phase protocol
    def select_targets(self, actions: np.ndarray) -> np.ndarray:
        """Candidate mask (n, n) for actions needing an explicit target
        (INVADE, and STRIKE with explicit coordinates); rows without a request
        are all-False. v2.3: strikes honor explicit target coordinates."""
        actions = np.asarray(actions, dtype=np.int64)
        need = (actions == A_INVADE) | (actions == A_STRIKE)
        return need[:, None] & self.known & self.alive[None, :] & (~self.in_domain)[None, :]

    def begin_tick(self, actions: np.ndarray, targets: np.ndarray | None):
        """Lock in actions + targets; apply invasion notices; return the set of
        invaded civilizations (they re-choose with full knowledge)."""
        self._actions = np.asarray(actions, dtype=np.int64)
        self._targets = {}
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int64)
            for i in np.where((self._actions == A_INVADE) | (self._actions == A_STRIKE))[0]:
                j = int(targets[i])
                if j >= 0 and self.select_targets(self._actions)[i, j]:
                    self._targets[int(i)] = j
        # invasion notices: the victim discovers the invader NOW
        self._invaders = {}
        for i, j in self._targets.items():
            if self._actions[i] == A_INVADE and self.alive[i] and self.alive[j]:
                self.known[j, i] = True
                self.first_detect_tick[j, i] = self.t
                self._invaders[int(j)] = int(i)
        return sorted(self._invaders.keys())

    def resolve(self, re_actions: np.ndarray):
        """Full tick resolution. `re_actions` overrides actions of invaded civs."""
        actions = self._actions.copy()
        re = np.asarray(re_actions, dtype=np.int64)
        for v in self._invaders:
            actions[v] = re[v]
        n = self.n
        rewards = np.zeros(n, dtype=np.float64)
        detections: list[tuple[int, int]] = []
        strikes: list[tuple[int, int, bool]] = []
        invasions: list[tuple[int, int, bool]] = []   # (invader, victim, mutual)
        eff = actions.copy()
        any_known = (self.known & self.alive[None, :]).any(axis=1)
        bad = (~any_known) & ((actions == A_STRIKE) | (actions == A_STRIKE_WEAK)
                              | (actions == A_STRIKE_STRONG) | (actions == A_INVADE)
                              | (actions == A_OBSERVE))
        eff[bad] = A_SILENT
        # in-domain / windup civs cannot use advanced actions
        eff[self.in_domain | self.domain_windup, ] = np.where(
            np.isin(eff[self.in_domain | self.domain_windup], [A_DEVELOP, A_BDOMAIN, A_SILENT]),
            eff[self.in_domain | self.domain_windup], A_SILENT) if (self.in_domain | self.domain_windup).any() else eff[self.in_domain | self.domain_windup]
        # tech gates (defensive; masks normally prevent these)
        eff[(eff == A_STRIKE) & ~(self.tech > self.gate_strike_tech)] = A_SILENT
        eff[(eff == A_STRIKE_WEAK) & ~(self.tech > self.gate_strike_tech)] = A_SILENT
        eff[(eff == A_STRIKE_STRONG) & ~(self.tech > self.gate_strike_tech)] = A_SILENT
        eff[(eff == A_OBSERVE) & ~(self.tech > self.gate_observe_tech)] = A_SILENT
        eff[(eff == A_BDOMAIN) & ~(self.tech > self.gate_domain_tech)] = A_SILENT
        gate_b = ~((self.tech > self.gate_broadcast_tech) | (self.res > self.gate_broadcast_res))
        eff[(eff == A_BSELF) & gate_b] = A_SILENT
        eff[(eff == A_BOTHER) & gate_b] = A_SILENT
        if not self.enable_black_domain:
            eff[actions == A_BDOMAIN] = A_SILENT

        # 0) domain activation (declared last tick -> active now, irreversible)
        self.in_domain |= self.domain_windup
        self.domain_windup[:] = False
        entering = (eff == A_BDOMAIN) & self.alive & ~self.in_domain & ~self.domain_windup
        rewards[entering] -= self.bd_entry_penalty          # instantaneous penalty
        self.domain_windup |= entering                       # active NEXT tick

        # 0.5) counter-detection of observers (v1.4 mechanism moved to OBSERVE)
        n_counter = 0
        if self.pending_counter:
            for i in self.pending_counter:
                if not self.alive[i] or self.in_domain[i]:
                    continue
                cand = np.where(self.alive & ~self.in_domain)[0]
                cand = cand[cand != i]
                if cand.size == 0:
                    continue
                k = int(self.rng.choice(cand))
                if not self.known[k, i]:
                    self.known[k, i] = True
                    self.first_detect_tick[k, i] = self.t
                    detections.append((k, i))
                    n_counter += 1
            self.pending_counter = []

        # 1) delayed broadcasts resolve — unconditionally: the signal was already
        #    sent, it does not care whether the sender (or the target) still exists
        for kind, j in list(self.pending_reveal):
            self.known[self.alive, j] = True
            self.public[j] = True
        self.pending_reveal = []

        # 2) passive detection (stealth depends on the target's action this tick)
        stealth = (eff[None, :] == A_SILENT)
        p_det = self.p_detect_base * np.sqrt(self.tech)[:, None]
        p_det = np.minimum(p_det * np.where(stealth, self.stealth_mult, 1.0), self.p_detect_max)
        p_det[self.in_domain | self.domain_windup, :] = 0.0
        p_det[:, self.in_domain] = 0.0
        dist = self._pair_dist()
        mask = (self.alive[:, None] & self.alive[None, :] & ~self.known
                & (dist <= self.detect_radius) & ~np.eye(n, dtype=bool))
        new_det = mask & (self.rng.uniform(size=(n, n)) < p_det)
        if new_det.any():
            ii, jj = np.where(new_det)
            for i, j in zip(ii, jj):
                self.known[i, j] = True
                self.first_detect_tick[i, j] = self.t
                detections.append((int(i), int(j)))

        # 3) broadcasts declared THIS tick are queued for next tick. v2.3: only
        #    ALL-broadcast (BROADCAST_SELF) exists — the sender publishes its own
        #    coordinates to everyone; the sender trivially knows it is exposed
        #    (its own action) and its public flag is visible to every observer.
        bself = (eff == A_BSELF) & self.alive
        for i in np.where(bself)[0]:
            self.pending_reveal.append(("self", int(i)))

        # 4) strikes (tech>7, immediate; v2.3: THREE target-selection modes over
        #    known coordinates — STRIKE=explicit target if given else nearest,
        #    STRIKE_WEAK=min tech known, STRIKE_STRONG=max tech known)
        #    SHIELD: a target with tech >= 2x the striker's tech BLOCKS the strike;
        #    blocking drains a large share of the TARGET's own resources into the
        #    free pool (the defender pays for its shield).
        known_alive_d = np.where(self.known & self.alive[None, :] & (~self.in_domain)[None, :]
                                 & (~self.domain_windup)[None, :], dist, np.inf)
        np.fill_diagonal(known_alive_d, np.inf)
        known_alive_tech = np.where(self.known & self.alive[None, :] & (~self.in_domain)[None, :]
                                    & (~self.domain_windup)[None, :], self.tech, np.inf)
        known_alive_tech_n = np.where(self.known & self.alive[None, :] & (~self.in_domain)[None, :]
                                      & (~self.domain_windup)[None, :], self.tech, -np.inf)
        np.fill_diagonal(known_alive_tech, np.inf)
        np.fill_diagonal(known_alive_tech_n, -np.inf)
        strike_rows = np.where(np.isin(eff, [A_STRIKE, A_STRIKE_WEAK, A_STRIKE_STRONG]) & self.alive)[0]
        kills: dict[int, int] = {}
        strikes_blocked: list[tuple[int, int, float]] = []
        strike_events: list[tuple] = []
        for i in strike_rows:
            mode = int(eff[i])
            if mode == A_STRIKE_WEAK:
                j = int(np.argmin(known_alive_tech[i]))
            elif mode == A_STRIKE_STRONG:
                j = int(np.argmax(known_alive_tech_n[i]))
            else:
                j = int(self._targets.get(int(i), -1))
                if j < 0 or not (self.known[i, j] and self.alive[j]) or self.in_domain[j] or self.domain_windup[j]:
                    j = int(np.argmin(known_alive_d[i]))
            if not np.isfinite(known_alive_d[i, j]):
                continue
            tech_i, tech_j = float(self.tech[i]), float(self.tech[j])
            if tech_j >= 2.0 * tech_i:
                shield_take = self.shield_cost_frac * max(self.res[j], 0.0)
                self.res[j] -= shield_take
                self.free_pool += shield_take
                strikes_blocked.append((int(i), int(j), float(shield_take)))
                strike_events.append((int(i), int(j), "blocked", tech_i, tech_j, float(shield_take)))
                continue
            p_kill = float(np.clip(self.kill_base + self.kill_tech_slope * (tech_i - tech_j), 0.15, 0.95))
            killed = bool(self.rng.uniform() < p_kill)
            strikes.append((int(i), int(j), killed))
            strike_events.append((int(i), int(j), "killed" if killed else "survived", tech_i, tech_j, 0.0))
            if killed and self.alive[j] and j not in kills:
                kills[int(j)] = int(i)

        # FATAL BUG FIX (2026-09-22): v2 regression — strike kills were recorded
        # (strike_events/kills dict) but the victim's alive flag was never
        # cleared, so struck civilizations stayed in the world. Mirrors the
        # invasion path below; corpse matter is reclaimed by the conservation
        # settlement (8b) and the death penalty applies via dead_now.
        for v, _killer in kills.items():
            if self.alive[v]:
                self.alive[v] = False
                self.death_cause[v] = "strike"

        # 5) invasions (guaranteed kill; victim re-chosen action may counter-invade)
        invasion_kill: dict[int, int] = {}
        for victim, invader in self._invaders.items():
            if not (self.alive[victim] and self.alive[invader]):
                continue
            if self.in_domain[victim] or self.domain_windup[victim]:
                continue   # cannot cross an (activating) black domain boundary
            re_v = int(actions[victim])
            counter = re_v == A_INVADE and self.alive[invader] and invader not in invasion_kill
            invasion_kill[int(victim)] = int(invader)
            invasions.append((int(invader), int(victim), bool(counter)))
            if counter:
                # mutual destruction: the victim drags the invader down with it
                if invader not in invasion_kill and self.alive[invader]:
                    invasion_kill[int(invader)] = int(victim)
                    invasions.append((int(victim), int(invader), True))
        for v in invasion_kill:
            if self.alive[v]:
                self.alive[v] = False
        # loot: half the victim's resources to each surviving killer; the rest of
        # every corpse (and dead invaders' estates) returns to the free pool
        loot_to = np.zeros(n)
        for victim, killer in invasion_kill.items():
            if self.alive[killer] and not self.alive[victim]:
                take = 0.5 * max(self.res[victim], 0.0)
                self.res[killer] += take
                loot_to[killer] += take
                self.res[victim] = 0.5 * max(self.res[victim], 0.0)

        dead_now = np.zeros(n, dtype=bool)
        for v in list(kills) + list(invasion_kill):
            if not self.alive[v]:
                dead_now[v] = True
        for v in kills:
            self.death_cause[v] = "strike"
        for victim, killer in invasion_kill.items():
            self.death_cause[victim] = "invade"
        for (v, _k, mutual) in invasions:
            if mutual:
                self.death_cause[v] = "mutual"

        # 6) observe (tech>5, immediate: intel now, exposure next tick)
        obs_rows = np.where((eff == A_OBSERVE) & self.alive)[0]
        for i in obs_rows:
            j = int(np.argmin(known_alive_d[i]))
            if np.isfinite(known_alive_d[i, j]):
                self.observed[i, j] = True
                self.pending_counter.append(int(i))   # to see is to be seen

        # 7) growth / explosion — v2.3: growth follows the PRE-v2.2 mechanism
        #    (develop_rate × (1 − tech/cap)); the cap is held at a huge value so
        #    tech stays unbounded and growth is effectively linear. The "steep
        #    early, flat late" shaping lives in the REWARD (Softsign), not growth.
        dev = (eff == A_DEVELOP) & self.alive
        growth = self.develop_rate * (1.0 - self.tech[dev] / self.tech_cap)
        if dev.any():
            growth = growth * np.where(self.in_domain[dev], self.bd_growth_mult, 1.0)
        self.tech[dev] += growth
        self.peak_tech = np.where(self.alive, np.maximum(self.peak_tech, self.tech), self.peak_tech)
        p_boom = np.full(n, self.p_explosion)
        p_boom[self.in_domain] *= self.bd_explosion_mult
        boom = self.alive & (self.rng.uniform(size=n) < p_boom)
        if self.p_explosion > 0 and boom.any():
            self.tech[boom] *= self.rng.uniform(1.6, 2.2, size=boom.sum())   # unbounded now

        # 8) resource depletion -> matter returns to the free pool; starvation if empty
        deplete = self.alive & (self.rng.uniform(size=n) < self.p_deplete)
        if deplete.any():
            loss = self.rng.uniform(self.deplete_lo, self.deplete_hi, size=deplete.sum())
            loss = np.minimum(loss, np.maximum(self.res[deplete], 0.0))      # cannot go below 0 twice
            self.res[deplete] -= loss
            self.free_pool += float(np.sum(loss))
        starved = self.alive & (self.res <= 0)
        if starved.any():
            self.alive[starved] = False
            for v in np.where(starved)[0]:
                self.death_cause[v] = "starve"
            dead_now |= starved

        # 8b) conservation settlement: every corpse's remaining matter returns to
        # the free pool (Axiom 2: total is constant; loot already moved above)
        if dead_now.any():
            for v in np.where(dead_now)[0]:
                self.free_pool += max(self.res[v], 0.0)
                self.res[v] = 0.0

        # 9) rewards: v2.2 (user clarification) — the tech-derived reward is the
        # Softsign variant of the agent's CURRENT tech: r_dev = scale·tech/(1+a·tech).
        # scale defaults 1.0 (user spec, per-tick bound 5 < death 10); the audit
        # variant v2.2s sets scale=0.01 (per-tick cap 0.05 = legacy develop_reward,
        # restoring return-level dominance of the death penalty).
        rewards[dead_now] -= self.death_penalty
        dev_alive = (eff == A_DEVELOP) & self.alive
        if dev_alive.any():
            rewards[dev_alive] += self.softsign_reward_scale * self.tech[dev_alive] / (
                1.0 + self.softsign_a * np.abs(self.tech[dev_alive]))

        # 10) staggered spawns: cohorts due now wake up (act from next tick)
        self._activate_pending_spawns()

        self.last_action = eff
        self.t += 1
        obs, valid = self._obs(), self._valid()
        # v2.3 settlement: fixed tick length is the world clock — the episode ends
        # at t_max; early termination only on physical heat death (everyone dead
        # AND the pool cannot afford the next pre-rolled civilization)
        done = bool(self.t >= self.t_max or
                    ((~self.alive).all() and self.free_pool < self.pending_spawns[0]))
        info = {
            "detections": detections, "strikes": strikes, "invasions": invasions,
            "loot_to": loot_to,
            "died_this_tick": [(int(v), self.death_cause[v]) for v in np.where(dead_now)[0]],
            "strikes_blocked": strikes_blocked,
            "strike_events": strike_events,
            "free_pool": float(self.free_pool),
            "res_conservation_residual": float(self.res_total - self.res.sum() - self.free_pool),
            "n_births": len(self.birth_log),
            "n_alive": int(self.alive.sum()), "in_domain": int(self.in_domain.sum()),
            "counter_detections": n_counter,
        }
        return obs, valid, rewards, self.alive.copy(), done, info, dead_now
