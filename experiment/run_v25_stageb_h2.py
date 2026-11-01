#!/usr/bin/env python3
"""v2.5 Stage B with v2b heuristics: re-run all four reward arms' mixed-pool
evaluation under the UPDATED heuristics (df counter-bite, pacifist hiding).
Idempotent per arm; merges into results/v25_arms_h2/summary_h2.json."""
import json, os, subprocess, sys, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ARMS = [
    ("a_classic", {}),
    ("b_terminal", {"softsign_reward_scale": 0.0, "bd_entry_penalty": 0.0}),
    ("c_res", {"res_reward_scale": 0.0025}),
    ("d_malintent", {"res_reward_scale": 0.0025, "attack_reward": 0.5}),
]

def main():
    merged = {}
    for tag, ov in ARMS:
        out = os.path.join(RUN, "results", "v25_arms_h2", tag)
        sb = os.path.join(out, "stage_b.json")
        if os.path.exists(sb):
            print(f"[{tag}] h2 stage_b skip (exists)", flush=True)
        else:
            base = yaml.safe_load(open(os.path.join(HERE, "config_v25.yaml")))
            base["env"].update(ov)
            cfgp = os.path.join(RUN, "runs", "v25_arms", tag, "config_h2.yaml")
            yaml.safe_dump(base, open(cfgp, "w"))
            r = subprocess.run(
                [sys.executable, "stage_b_v25b.py", "--run-dir",
                 os.path.join(RUN, "runs", "v25_arms", tag, "run", "seed42_main"),
                 "--episodes", "150", "--out", out, "--config", cfgp],
                env={**os.environ, "OMP_NUM_THREADS": "1"}, cwd=HERE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(f"[{tag}] STAGEB_V2B_EXIT={r.returncode}", flush=True)
            if r.returncode != 0:
                print(r.stdout[-800:], flush=True)
                continue
        d = json.load(open(sb))
        merged[tag] = {"override": ov,
                       "survival_by_kind": d["survival_by_kind"],
                       "learner_action_profile": d["learner_action_profile"],
                       "learner_loot_mean": d.get("learner_loot_mean"),
                       "learner_kills_mean": d.get("learner_kills_mean"),
                       "learner_invades_per_episode": d.get("learner_invades_per_episode"),
                       "survival_by_earned_tech_class": d.get("survival_by_earned_tech_class")}
        print(f"[{tag}] surv:", merged[tag]["survival_by_kind"], flush=True)
    outp = os.path.join(RUN, "results", "v25_arms_h2", "summary_h2.json")
    json.dump(merged, open(outp, "w"), indent=2, ensure_ascii=False)
    print(f"-> {outp} ({len(merged)}/4 arms)", flush=True)

if __name__ == "__main__":
    main()
