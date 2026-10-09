"""DarkForestGrid v1.3 — a minimal multi-agent environment encoding the axioms of
the "Dark Forest" theory from Liu Cixin's *Three-Body* trilogy.

Axiom -> mechanism mapping (full ladder in experiment_design.md §2/§2.1/§2.2/§2.3):
  A1  survival is the first need   -> reward +1.0 per tick alive; death is terminal
  A2  civilization grows, matter
      is finite                    -> v1.2 territorial crowding (growth divided by
                                      alive neighbours within detect_radius) +
                                      v1.3 survival decay (tech *= 1-decay each tick;
                                      below collapse_threshold the civilization dies)
                                      -> growth is EXISTENTIAL, not optional
  M1  chain of suspicion           -> intentions unobservable; only anonymous
                                      coordinate broadcast; known star = known forever
  M2  technological explosion      -> p=0.01/tick jump tech *= U[1.6, 2.2] (uncapped)
  DF corollary (hide + strike)     -> exists as ACTIONS only; never imposed

v1.3 additions (co-designed with the user, 2026-09-05):
  * strike_penalty = 0 (main): "we merely awaken other forces of the universe" —
    strikes are free for the attacker (only the small tech *0.96 logistics cost)
  * BLACK_DOMAIN action (6th): lock yourself into a reduced-lightspeed domain:
      - cannot be discovered and cannot discover (even broadcast cannot reveal you)
      - instantaneous entry penalty (bd_entry_penalty=0.5)
      - growth x0.2 inside (DEVELOP stays inside), explosion probability x0.1
      - broadcasting and striking forbidden inside
      - exit = choose SILENT inside (unseal: still protected+blind that tick)
      - pre-existing knowledge still allows memory strikes (the star was seen before)
      - survival decay still applies inside (camping forever is not viable)

Version ladder (same codebase, config-gated):
  v1.0  no competition channel                        -> peaceful developer (pilot)
  v1.1  + finite global matter pool (tech capped 10)  -> still peace (saturation)
  v1.2  + uncapped tech + territorial crowding        -> still peace (no existential
         pressure: in a peaceful world tech is worth nothing to defend)
  v1.3  + survival decay + free strikes + black domain -> THIS configuration

Action space (6): 0 SILENT, 1 DEVELOP, 2 BROADCAST_SELF, 3 BROADCAST_OTHER, 4 STRIKE,
5 BLACK_DOMAIN. STRIKE/BROADCAST_OTHER valid iff the actor knows >=1 alive civilization.
"""
from __future__ import annotations

import numpy as np

ACTION_NAMES = ["SILENT", "DEVELOP", "BROADCAST_SELF", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN"]
A_SILENT, A_DEVELOP, A_BSELF, A_BOTHER, A_STRIKE, A_BDOMAIN = range(6)

OBS_DIM = 28
N_ACTIONS = 6


class DarkForestEnv:
    """Vectorised multi-agent Dark Forest environment (numpy)."""

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
        kill_base: float = 0.55,
        kill_tech_slope: float = 0.12,
        strike_cost_mult: float = 0.96,
        strike_penalty: float = 0.25,
        develop_reward: float = 0.05,
        alive_reward: float = 1.0,
        min_spawn_dist: float = 12.0,
        explosion_lo: float = 1.6,
        explosion_hi: float = 2.2,
        matter_total: float = 400.0,
        develop_cost: float = 0.25,
        passive_cost: float = 0.02,
        neighbor_suppression: float = 0.0,
        decay_rate: float = 0.025,
        collapse_threshold: float = 0.5,
        enable_black_domain: bool = True,
        bd_entry_penalty: float = 0.5,
        bd_growth_mult: float = 0.2,
        bd_explosion_mult: float = 0.1,
        enable_counter_detection: bool = False,
        terminal_survival_reward: float = 0.0,
        seed: int = 0,
    ):
        self.n = n_agents
        self.world = float(world)
        self.detect_radius = float(detect_radius)
        self.t_max = int(t_max)
        self.p_detect_base = p_detect_base
        self.stealth_mult = stealth_mult
        self.p_detect_max = p_detect_max
        self.p_explosion = p_explosion
        self.develop_rate = develop_rate
        self.tech_cap = tech_cap
        self.kill_base = kill_base
        self.kill_tech_slope = kill_tech_slope
        self.strike_cost_mult = strike_cost_mult
        self.strike_penalty = strike_penalty
        self.develop_reward = develop_reward
        self.alive_reward = alive_reward
        self.min_spawn_dist = min_spawn_dist
        self.explosion_lo = explosion_lo
        self.explosion_hi = explosion_hi
        self.matter_total = float(matter_total)
        self.develop_cost = float(develop_cost)
        self.passive_cost = float(passive_cost)
        self.neighbor_suppression = float(neighbor_suppression)
        self.decay_rate = float(decay_rate)
        self.collapse_threshold = float(collapse_threshold)
        self.enable_black_domain = bool(enable_black_domain)
        self.bd_entry_penalty = float(bd_entry_penalty)
        self.bd_growth_mult = float(bd_growth_mult)
        self.bd_explosion_mult = float(bd_explosion_mult)
        self.enable_counter_detection = bool(enable_counter_detection)
        self.terminal_survival_reward = float(terminal_survival_reward)
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    # ------------------------------------------------------------------ utils
    def _torus_delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a[None, :, :] - b[:, None, :]          # (n, n, 2)
        d -= self.world * np.round(d / self.world)
        return d

    def _pair_dist(self, pts: np.ndarray | None = None) -> np.ndarray:
        pts = self.pos if pts is None else pts
        d = self._torus_delta(pts, pts)
        return np.linalg.norm(d, axis=-1)          # (n, n)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None):
        """Reseed with an explicit seed, or draw a fresh deterministic seed from
        the master stream so that consecutive episodes differ but the whole
        sequence stays reproducible from the constructor seed."""
        if seed is None:
            seed = int(self.rng.integers(1 << 31))
        self.rng = np.random.default_rng(seed)
        n = self.n
        while True:
            pos = self.rng.uniform(0, self.world, size=(n, 2))
            d = self._pair_dist(pos) + np.eye(n) * 1e9
            if d.min() >= self.min_spawn_dist:
                break
        self.pos = pos
        self.tech = np.ones(n, dtype=np.float64)
        self.alive = np.ones(n, dtype=bool)
        self.known = np.zeros((n, n), dtype=bool)      # known[i,j]: i knows j's position
        self.public = np.zeros(n, dtype=bool)          # public[j]: known to everyone
        self.last_action = np.full(n, A_SILENT, dtype=np.int64)
        self.first_detect_tick = -np.ones((n, n), dtype=np.int64)
        self.in_domain = np.zeros(n, dtype=bool)   # black-domain state (irreversible in v1.4)
        self.pending_counter: list[int] = []       # detectors awaiting guaranteed exposure
        self.matter = float(self.matter_total)
        self.t = 0
        return self._obs(), self._valid()

    # ------------------------------------------------------------------ masks
    def _valid(self) -> np.ndarray:
        # strike / broadcast-other need at least one known alive civilization that
        # is NOT shielded inside a black domain (v1.4: the domain is absolute)
        strikeable = (self.known & self.alive[None, :] & (~self.in_domain)[None, :]).any(axis=1)
        any_known = strikeable & self.alive
        valid = np.ones((self.n, N_ACTIONS), dtype=bool)
        valid[:, A_BOTHER] = any_known
        valid[:, A_STRIKE] = any_known
        valid[:, A_BDOMAIN] = self.enable_black_domain
        # inside the black domain: no broadcasting, no striking; the domain cannot
        # be revoked (SILENT inside is mere stasis, NOT an exit)
        valid[self.in_domain, A_BOTHER] = False
        valid[self.in_domain, A_STRIKE] = False
        valid[self.in_domain, A_BSELF] = False
        return valid

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        n = self.n
        dist = self._pair_dist()
        alive_idx = self.alive
        known_alive = self.known & alive_idx[None, :]
        np.fill_diagonal(known_alive, False)
        n_known = known_alive.sum(axis=1).astype(np.float64)
        n_alive = alive_idx.sum()
        crowd = ((dist <= self.detect_radius) & alive_idx[None, :]
                 & ~np.eye(n, dtype=bool)).sum(axis=1).astype(np.float64)

        # 3 nearest known-alive per agent (fully vectorised)
        big = dist.copy()
        big[~known_alive] = np.inf
        order = np.argsort(big, axis=1)[:, :3]                    # (n, 3)
        near_d = np.take_along_axis(dist, order, axis=1)          # (n, 3)
        near_known = np.isfinite(near_d)
        near_d = np.where(near_known, near_d, 0.0)
        near_pos = self.pos[order]                                # (n, 3, 2)
        delta = near_pos - self.pos[:, None, :]
        delta -= self.world * np.round(delta / self.world)
        near_tech = self.tech[order]
        near_pub = self.public[order].astype(np.float64)

        onehot = np.zeros((n, N_ACTIONS), dtype=np.float64)
        onehot[np.arange(n), self.last_action] = 1.0
        indomain = self.in_domain.astype(np.float64)[:, None]

        obs = np.concatenate(
            [
                (self.tech / self.tech_cap)[:, None],
                (n_known / (n - 1))[:, None],
                np.full((n, 1), n_alive / n),
                np.full((n, 1), self.t / self.t_max),
                np.full((n, 1), self.matter / self.matter_total),
                (crowd / (n - 1))[:, None],
                onehot,
                indomain,
                (delta[..., 0] / 50.0),
                (delta[..., 1] / 50.0),
                (near_d / 50.0),
                ((near_tech - self.tech[:, None]) / 5.0),
                near_pub,
            ],
            axis=1,
        ).astype(np.float32)
        obs[~alive_idx] = 0.0
        return obs

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        n = self.n
        rewards = np.zeros(n, dtype=np.float64)
        detections: list[tuple[int, int]] = []
        strikes: list[tuple[int, int, bool]] = []

        dist = self._pair_dist()
        any_known = (self.known & self.alive[None, :]).any(axis=1)
        eff = actions.copy()
        # defensive fallbacks (policies normally mask invalid actions)
        bad = (~any_known) & ((actions == A_BOTHER) | (actions == A_STRIKE))
        eff[bad] = A_SILENT
        # inside the domain: broadcast/strike are forbidden (policies mask these;
        # the fallback keeps the env correct even against unmasked inputs)
        eff[self.in_domain & ((actions == A_BOTHER) | (actions == A_STRIKE) | (actions == A_BSELF))] = A_SILENT
        if not self.enable_black_domain:
            eff[actions == A_BDOMAIN] = A_SILENT

        # 0) black-domain state machine (v1.4: IRREVERSIBLE absolute sanctuary)
        #    enter: choose BLACK_DOMAIN from outside (instantaneous penalty)
        #    inside: DEVELOP = slow growth (x0.2); SILENT/BLACK_DOMAIN = stasis;
        #            no broadcasting, no striking; immune to ALL strikes incl.
        #            memory strikes; undetectable; decay still applies (camping
        #            forever is not viable)
        entering = (eff == A_BDOMAIN) & self.alive & ~self.in_domain
        rewards[entering] -= self.bd_entry_penalty
        self.in_domain |= entering

        # 0.5) counter-detection (v1.4, "to see is to be seen"): a civilization
        #      that passively discovered someone last tick is NOW, after its first
        #      post-discovery decision, guaranteed to be discovered by a random
        #      other alive civilization. Entering the black domain dodges it (the
        #      domain is absolute); dead observers drop; no cascade (the guaranteed
        #      counter-detection is the universe's symmetry tax, paid once).
        n_counter = 0
        if self.enable_counter_detection and self.pending_counter:
            for i in self.pending_counter:
                if not self.alive[i] or self.in_domain[i]:
                    continue                      # dead detectors reveal nothing; domain dodges
                candidates = np.where(self.alive & ~self.in_domain)[0]
                candidates = candidates[candidates != i]
                if candidates.size == 0:
                    continue
                k = int(self.rng.choice(candidates))
                if not self.known[k, i]:
                    self.known[k, i] = True
                    self.first_detect_tick[k, i] = self.t
                    detections.append((k, i))
                    n_counter += 1
            self.pending_counter = []

        rewards[self.alive] += self.alive_reward
        rewards[(eff == A_DEVELOP) & self.alive] += self.develop_reward
        rewards[(eff == A_STRIKE) & self.alive] -= self.strike_penalty
        self.last_action = eff

        # 1) detection — in-domain civilizations cannot discover and cannot be
        #    discovered (even via broadcast); stealth depends on the target's
        #    action *this* tick
        stealth = (eff[None, :] == A_SILENT)                      # (n, n): column j silent
        p_det = self.p_detect_base * np.sqrt(self.tech)[:, None]
        p_det = np.minimum(p_det * np.where(stealth, self.stealth_mult, 1.0), self.p_detect_max)
        p_det[self.in_domain, :] = 0.0
        p_det[:, self.in_domain] = 0.0
        mask = (
            self.alive[:, None]
            & self.alive[None, :]
            & ~self.known
            & (dist <= self.detect_radius)
            & ~np.eye(n, dtype=bool)
        )
        new_det = mask & (self.rng.uniform(size=(n, n)) < p_det)
        if new_det.any():
            ii, jj = np.where(new_det)
            for i, j in zip(ii, jj):
                self.known[i, j] = True
                self.first_detect_tick[i, j] = self.t
                detections.append((int(i), int(j)))
                if self.enable_counter_detection:
                    self.pending_counter.append(int(i))   # to see is to be seen

        # 2) broadcasts (anonymous, global); BROADCAST_OTHER targets the highest-tech
        #    known alive civilization; in-domain civilizations cannot be revealed
        bself = (eff == A_BSELF) & self.alive
        if bself.any():
            for i in np.where(bself)[0]:
                self.known[self.alive, i] = True
                self.public[i] = True
        bother = (eff == A_BOTHER) & self.alive & any_known
        known_alive_tech = np.where(
            self.known & self.alive[None, :] & (~self.in_domain)[None, :],
            self.tech[None, :], -np.inf)
        np.fill_diagonal(known_alive_tech, -np.inf)
        for i in np.where(bother)[0]:
            j = int(np.argmax(known_alive_tech[i]))
            if np.isfinite(known_alive_tech[i, j]):
                self.known[self.alive, j] = True
                self.public[j] = True

        # 3) strikes — targets precomputed, then resolved simultaneously; anonymous.
        #    v1.4: civilizations inside a black domain are ABSOLUTE sanctuaries —
        #    not even memory strikes (pre-existing knowledge) can reach them.
        strike_rows = np.where((eff == A_STRIKE) & self.alive & any_known)[0]
        known_alive_d = np.where(
            self.known & self.alive[None, :] & (~self.in_domain)[None, :],
            dist, np.inf)
        np.fill_diagonal(known_alive_d, np.inf)
        targets = {int(i): int(np.argmin(known_alive_d[i])) for i in strike_rows}
        targets = {i: j for i, j in targets.items() if np.isfinite(known_alive_d[i, j])}
        kills: dict[int, int] = {}
        for i, j in targets.items():
            p_kill = float(np.clip(self.kill_base + self.kill_tech_slope * (self.tech[i] - self.tech[j]), 0.15, 0.95))
            killed = bool(self.rng.uniform() < p_kill)
            self.tech[i] *= self.strike_cost_mult
            if killed:
                kills[j] = i
            else:
                self.tech[j] *= 0.90
            strikes.append((int(i), int(j), killed))
        dead_now = np.zeros(n, dtype=bool)
        for j in kills:
            self.alive[j] = False
            dead_now[j] = True

        # 4) growth: DEVELOP yields growth, suppressed by territorial crowding
        #    (v1.2) and slowed 5x inside the black domain (v1.3). With
        #    develop_cost=0 / passive_cost=0 the v1.1 matter pool is inert.
        if self.passive_cost > 0:
            self.matter = max(0.0, self.matter - self.passive_cost * float(self.alive.sum()))
        dev = (eff == A_DEVELOP) & self.alive
        growth = self.develop_rate * (1.0 - self.tech[dev] / self.tech_cap)
        if self.neighbor_suppression > 0 and dev.any():
            crowd = ((dist <= self.detect_radius) & self.alive[None, :]
                     & ~np.eye(n, dtype=bool)).sum(axis=1)
            growth = growth / (1.0 + self.neighbor_suppression * crowd[dev])
        if dev.any():
            growth = growth * np.where(self.in_domain[dev], self.bd_growth_mult, 1.0)
        if self.develop_cost > 0:
            if dev.any():
                want = self.develop_cost * float(dev.sum())
                take = min(want, self.matter)
                growth = growth * (take / want)
                self.matter -= take
            else:
                growth = growth * 0.0
        self.tech[dev] += growth
        p_boom = np.full(n, self.p_explosion)
        p_boom[self.in_domain] *= self.bd_explosion_mult
        boom = self.alive & (self.rng.uniform(size=n) < p_boom)
        if self.p_explosion > 0 and boom.any():
            self.tech[boom] *= self.rng.uniform(self.explosion_lo, self.explosion_hi, size=boom.sum())
        self.tech = np.minimum(self.tech, self.tech_cap)

        # 5) survival decay (v1.3): growth is existential, not optional. Stagnant
        #    civilizations wither; below the collapse threshold a civilization dies.
        if self.decay_rate > 0:
            self.tech[self.alive] *= (1.0 - self.decay_rate)
        collapsing = self.alive & (self.tech < self.collapse_threshold)
        if collapsing.any():
            for j in np.where(collapsing)[0]:
                self.alive[j] = False
                dead_now[j] = True

        # v1.4: the black domain cannot be revoked — no unsealing exists.

        self.t += 1
        obs, valid = self._obs(), self._valid()
        done = bool((~self.alive).all() or self.t >= self.t_max)
        # terminal-only reward mode (v1.4 branch): surviving to the time limit pays
        # off; extinction pays nothing; all shaping rewards are zeroed via config
        if done and self.terminal_survival_reward > 0 and self.t >= self.t_max:
            rewards[self.alive] += self.terminal_survival_reward
        info = {
            "detections": detections,
            "strikes": strikes,
            "n_alive": int(self.alive.sum()),
            "in_domain": int(self.in_domain.sum()),
            "counter_detections": n_counter,
        }
        return obs, valid, rewards, self.alive.copy(), done, info, dead_now
