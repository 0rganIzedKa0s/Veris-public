"""Mode switching through the arbiter: planner only vs planner + gate votes.

  .venv/bin/python -m evals.switching_eval          # test seeds 0-4
  .venv/bin/python -m evals.switching_eval --dev    # dev seeds 100-104 (tuning)

Every sim/gps.py scenario x test seeds 0-4. The arbiter is driven at 1 Hz with
the offline planner's proposal (update_context) and ground-truth walking state
(update_motion; IMU walking-vs-still is assumed solved, see
docs/sensor-fusion-switching.md finding 5). Condition (b) adds simulated Tier 0
gate votes at 5 Hz (update_gate) derived from ground truth with a given error
rate. Errors are bursty (two-state Markov chain, mean burst 3 s), since a real
classifier is wrong for runs of similar frames, not independently per frame.
The same error process corrupts text_p (a wrong text vote reads 0.6-0.9 while
no text is present).

Metrics: time in the ground-truth mode; switch latency after each true mode
change; flaps (a switch reversed within 20 s); and the doorway case: seconds
from walking in until the arbiter leaves outdoor_nav, next to the GNSS lag the
simulator drew for that doorway.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys

from evals.planner_eval import DEV_SEEDS, TEST_SEEDS, _block_network, lag_after, summarize, transitions

OUT = Path(__file__).parent / "results" / "switching.json"
ERROR_RATES = (0.05, 0.15, 0.30)
VOTE_HZ = 5
BURST_S = 3.0
FLAP_S = 20.0
INDOOR_MODES = {"indoor_nav", "describe", "text_reading"}


def gate_votes(truth_indoor, err, seed):
    """Per tick, VOTE_HZ (indoor_p, text_p) pairs."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / VOTE_HZ
    p_exit = dt / BURST_S
    p_enter = p_exit * err / (1 - err)
    state = [False, False]                       # in an error burst? (indoor, text)
    out = []
    for ind in truth_indoor:
        tick = []
        for _ in range(VOTE_HZ):
            for s in range(2):
                state[s] = (rng.random() >= p_exit) if state[s] else (rng.random() < p_enter)
            say_in = ind != state[0]
            ip = rng.uniform(0.75, 0.95) if say_in else rng.uniform(0.05, 0.25)
            tp = rng.uniform(0.6, 0.9) if state[1] else rng.uniform(0.0, 0.15)
            tick.append((ip, tp))
        out.append(tick)
    return out


def drive(recs, proposals, votes=None):
    from veris.arbiter import Arbiter, Motion
    from veris.pack_loop import Clock

    clock = Clock(0.0)
    arb = Arbiter(now=clock)
    arb.user_command(recs[0]["truth"]["mode"])   # user starts in the right mode
    modes = []
    for k, (r, (mode, sit)) in enumerate(zip(recs, proposals)):
        clock.set(float(k))
        arb.update_motion(Motion.WALKING if r["truth"]["moving"] else Motion.STATIONARY)
        arb.update_context(mode, sit)
        if votes is not None:
            for j, (ip, tp) in enumerate(votes[k]):
                clock.set(k + j / VOTE_HZ)
                arb.update_gate(ip, tp)
        modes.append(arb.mode)
    return modes


def flaps(modes):
    sw = [(i, a, b) for i, a, b in transitions(modes)]
    return sum(1 for (t1, a1, b1), (t2, a2, b2) in zip(sw, sw[1:]) if t2 - t1 <= FLAP_S and b2 == a1)


def operating_points(w):
    """The latency/flap trade-off of the gate fusion knobs, on the DEV seeds
    (where the defaults were picked); the headline numbers above are test."""
    import veris.arbiter as A
    from sim import gps
    from veris.planner import Planner

    saved = (A.GATE_TAU_S, A.GPS_W, A.GATE_IN, A.GATE_OUT)
    runs = []
    for name in gps.SCENARIOS:
        for seed in DEV_SEEDS:
            recs = gps.simulate(name, seed)
            pl = Planner(index=w.ix)
            runs.append((recs, [(s.suggested_mode, s.situation) for s in (pl.plan(r["fix"]) for r in recs)]))
    pts = []
    for tau, gw, band in ((4, 0.1, 0.65), (4, 0.1, 0.7), (6, 0.1, 0.7), (6, 0.3, 0.65), (8, 0.3, 0.7)):
        A.GATE_TAU_S, A.GPS_W, A.GATE_IN, A.GATE_OUT = tau, gw, band, 1 - band
        for e in ERROR_RATES:
            fl, corr, door = 0, [], []
            for k, (recs, props) in enumerate(runs):
                T_in = [r["truth"]["indoor"] for r in recs]
                T = [r["truth"]["mode"] for r in recs]
                M = drive(recs, props, gate_votes(T_in, e, k))
                fl += flaps(M)
                corr.append(sum(a == b for a, b in zip(M, T)) / len(T))
                for i, a, b in transitions(T_in):
                    if b:
                        j = i
                        while j < len(M) and T_in[j] and M[j] not in INDOOR_MODES:
                            j += 1
                        door.append(j - i)
            pts.append({"tau_s": tau, "gps_w": gw, "band": [round(1 - band, 2), band], "err": e,
                        "correct": round(st.mean(corr), 3), "flaps": fl,
                        "doorway_median_s": st.median(door), "doorway_max_s": max(door)})
    A.GATE_TAU_S, A.GPS_W, A.GATE_IN, A.GATE_OUT = saved
    return {"seeds": DEV_SEEDS, "points": pts}


def main():
    dev = "--dev" in sys.argv[1:]
    seeds = DEV_SEEDS if dev else TEST_SEEDS
    out_path = OUT.with_name("switching_dev.json") if dev else OUT
    _block_network()
    from sim import gps
    from veris.planner import Planner

    w = gps.world()
    conds = ["planner_only"] + [f"planner+gate_err{e:.2f}" for e in ERROR_RATES]
    acc = defaultdict(lambda: defaultdict(list))      # cond -> scenario -> [frac]
    lat = defaultdict(lambda: defaultdict(list))      # cond -> transition -> [s]
    nflap = defaultdict(int)
    door = defaultdict(list)
    ticks = 0
    for name in gps.SCENARIOS:
        for seed in seeds:
            recs = gps.simulate(name, seed)
            ticks += len(recs)
            pl = Planner(index=w.ix)
            props = [(s.suggested_mode, s.situation) for s in (pl.plan(r["fix"]) for r in recs)]
            T = [r["truth"]["mode"] for r in recs]
            T_in = [r["truth"]["indoor"] for r in recs]
            for c in conds:
                votes = None
                if c != "planner_only":
                    e = float(c.split("err")[1])
                    votes = gate_votes(T_in, e, seed * 1000 + int(e * 100))
                M = drive(recs, props, votes)
                acc[c][name].append(sum(a == b for a, b in zip(M, T)) / len(T))
                for i, a, b in transitions(T):
                    lat[c][f"{a}->{b}"].append(lag_after(T, M, i))
                nflap[c] += flaps(M)
                for i, a, b in transitions(T_in):
                    if not b:
                        continue
                    g = 0
                    while i + g < len(recs) and recs[i + g]["truth"]["gnss_phase"] == "lag":
                        g += 1
                    j = i
                    while j < len(M) and T_in[j] and M[j] not in INDOOR_MODES:
                        j += 1
                    got = j < len(M) and T_in[j]
                    door[c].append({"scenario": name, "seed": seed, "gnss_lag_s": g,
                                    "arbiter_s": (j - i) if got else None})
    res = {}
    for c in conds:
        per = {n: round(st.mean(v), 3) for n, v in acc[c].items()}
        allv = [x for v in acc[c].values() for x in v]
        d = door[c]
        got = [x["arbiter_s"] for x in d if x["arbiter_s"] is not None]
        res[c] = {
            "time_in_correct_mode_mean_over_runs": round(st.mean(allv), 3),
            "time_in_correct_mode_per_scenario": per,
            "switch_latency_s": {k: summarize(v) for k, v in sorted(lat[c].items())},
            "flaps_reversed_within_20s": nflap[c],
            "flaps_per_hour": round(nflap[c] / (ticks / 3600), 2),
            "doorway": {
                "n": len(d),
                "median_arbiter_s": st.median(got) if got else None,
                "max_arbiter_s": max(got) if got else None,
                "missed": len(d) - len(got),
                "beat_gnss_lag": sum(1 for x in d if x["arbiter_s"] is not None and x["arbiter_s"] < x["gnss_lag_s"]),
                "median_gnss_lag_s": st.median(x["gnss_lag_s"] for x in d),
                "runs": d,
            },
        }
    out = {
        "operating_points_dev": operating_points(w) if not dev else None,
        "command": "python -m evals.switching_eval" + (" --dev" if dev else ""),
        "split": "dev" if dev else "test", "seeds": seeds, "N_scenarios": len(gps.SCENARIOS),
        "N_ticks_per_condition": ticks,
        "setup": {"tick_hz": 1, "gate_vote_hz": VOTE_HZ, "error_burst_mean_s": BURST_S,
                  "flap_window_s": FLAP_S, "motion": "ground truth walking/stationary",
                  "arbiter_start": "user_command(ground-truth mode at t=0)",
                  "ground_truth_mode": "outdoors outdoor_nav; indoors moving indoor_nav; indoors still describe",
                  "airgap": "socket.socket patched to raise"},
        "conditions": res,
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    for c in conds:
        r = res[c]
        print(f"{c:<24} correct={r['time_in_correct_mode_mean_over_runs']:.3f} flaps={r['flaps_reversed_within_20s']}"
              f" doorway median={r['doorway']['median_arbiter_s']}s (gnss {r['doorway']['median_gnss_lag_s']}s)"
              f" beat_gnss={r['doorway']['beat_gnss_lag']}/{r['doorway']['n']} missed={r['doorway']['missed']}")
    print("->", out_path)


if __name__ == "__main__":
    main()
