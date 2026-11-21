"""Offline planner vs simulator ground truth, with networking disabled.

  .venv/bin/python -m evals.planner_eval            # test seeds 0-4
  .venv/bin/python -m evals.planner_eval --dev      # dev seeds 100-104 (tuning)

Every scenario in sim/gps.py x every seed. The planner's thresholds were tuned
on the dev seeds only; the reported numbers come from the test seeds. The
cloud context engine's rules fallback (context_engine/composer.py, LLM off,
fed the same local-index facts) runs alongside for comparison; no paid API is
called and the whole run happens with socket.socket patched to raise.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "results"
TEST_SEEDS, DEV_SEEDS = [0, 1, 2, 3, 4], [100, 101, 102, 103, 104]
INDOOR_SITUATIONS = {"restaurant", "cafe", "retail_store", "office_building", "residential"}


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise OSError("network disabled for this eval")


def _block_network():
    socket.socket = _NoNet
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("network disabled"))


def _cloud_rules():
    """The cloud engine's rules fallback, offline, on facts from the local index."""
    from context_engine.composer import Composer
    from context_engine.maps import PlaceFacts

    comp = Composer(use_llm=False)
    assert not comp.use_llm

    def run(planner, fix):
        street, d = planner._street(fix["lat"], fix["lon"])
        nearby = [{"name": p["name"], "types": p["types"], "distance_m": round(p["distance_m"]),
                   "bearing_deg": round(p["bearing_deg"])} for p in planner._nearby(fix["lat"], fix["lon"])][:8]
        facts = PlaceFacts(lat=fix["lat"], lon=fix["lon"], street=street if d < 60 else None,
                           nearby=nearby, source="osm_local")
        res, _ = comp.compose(facts, fix.get("heading_deg"), fix.get("speed_mps"))
        return res["situation"], res["suggested_mode"]
    return run


def transitions(seq):
    """[(t, old, new)] where a per-tick label changes."""
    return [(i, a, b) for i, (a, b) in enumerate(zip(seq, seq[1:]), start=1) if a != b]


def lag_after(truth, pred, i0):
    """Seconds from a true change at tick i0 until pred first matches the new
    truth, within that truth segment; None if it never does."""
    j = i0
    while j < len(truth) and truth[j] == truth[i0]:
        if pred[j] == truth[i0]:
            return j - i0
        j += 1
    return None


def summarize(lags):
    got = [x for x in lags if x is not None]
    return {"n": len(lags), "missed": len(lags) - len(got),
            "median_s": st.median(got) if got else None,
            "mean_s": round(st.mean(got), 1) if got else None,
            "max_s": max(got) if got else None}


def run(seeds):
    from sim import gps
    from veris.planner import Planner

    w = gps.world()
    cloud = _cloud_rules()
    per_scn, ms = {}, []
    agg = defaultdict(lambda: defaultdict(list))
    for name in gps.SCENARIOS:
        route = gps.route_of(name)
        rows = defaultdict(list)
        for seed in seeds:
            recs = gps.simulate(name, seed)
            pl = Planner(index=w.ix)
            P_sit, P_mode, P_in, C_sit, C_mode, C_in = [], [], [], [], [], []
            csit = cmode = None
            for r in recs:
                t0 = time.perf_counter()
                s = pl.plan(r["fix"], route)
                ms.append((time.perf_counter() - t0) * 1000)
                P_sit.append(s.situation); P_mode.append(s.suggested_mode); P_in.append(pl.indoor)
                if r["fix"] is not None:
                    csit, cmode = cloud(pl, r["fix"])
                C_sit.append(csit); C_mode.append(cmode); C_in.append(csit in INDOOR_SITUATIONS)
            T_sit = [r["truth"]["situation"] for r in recs]
            T_mode = [r["truth"]["mode"] for r in recs]
            T_in = [r["truth"]["indoor"] for r in recs]
            for who, sit, mode, ind in (("planner", P_sit, P_mode, P_in), ("cloud_rules", C_sit, C_mode, C_in)):
                n = len(recs)
                rows[f"{who}.situation_acc"].append(sum(a == b for a, b in zip(sit, T_sit)) / n)
                rows[f"{who}.indoor_acc"].append(sum(a == b for a, b in zip(ind, T_in)) / n)
                rows[f"{who}.mode_acc"].append(sum(a == b for a, b in zip(mode, T_mode)) / n)
                for i, a, b in transitions(T_sit):
                    agg[who][f"situation {a}->{b}"].append(lag_after(T_sit, sit, i))
                for i, a, b in transitions(T_in):
                    agg[who]["indoor " + ("out->in" if b else "in->out")].append(lag_after(T_in, ind, i))
            for i, a, b in transitions(T_in):
                if b:   # the sampled GNSS doorway lag, for reference
                    j = i
                    while j < len(recs) and recs[j]["truth"]["gnss_phase"] == "lag":
                        j += 1
                    agg["sim"]["gnss_doorway_lag_s"].append(j - i)
            rows["ticks"].append(len(recs))
        per_scn[name] = {k: (sum(v) if k == "ticks" else round(st.mean(v), 3)) for k, v in rows.items()}
    overall = {}
    for k in next(iter(per_scn.values())):
        if k != "ticks":
            num = sum(per_scn[n][k] * per_scn[n]["ticks"] for n in per_scn)
            overall[k] = round(num / sum(per_scn[n]["ticks"] for n in per_scn), 3)
    lags = {who: {k: summarize(v) for k, v in sorted(d.items())} for who, d in agg.items() if who != "sim"}
    return {
        "per_scenario": per_scn,
        "overall_tick_weighted": overall,
        "transition_lag_s": lags,
        "gnss_doorway_lag_s_sampled": agg["sim"]["gnss_doorway_lag_s"],
        "ms_per_fix": {"mean": round(st.mean(ms), 3), "p95": round(sorted(ms)[int(0.95 * len(ms))], 3),
                       "max": round(max(ms), 3), "n": len(ms)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    seeds = DEV_SEEDS if a.dev else TEST_SEEDS
    _block_network()
    try:                                   # prove the block is live
        socket.create_connection(("1.1.1.1", 80), timeout=1)
        blocked = False
    except OSError:
        blocked = True
    res = run(seeds)
    import veris.planner as vp
    net_mods = [m for m in ("urllib.request", "http.client", "requests", "httpx", "aiohttp")
                if m in vp.__dict__]
    out = {
        "command": "python -m evals.planner_eval" + (" --dev" if a.dev else ""),
        "split": "dev" if a.dev else "test",
        "seeds": seeds,
        "N_scenarios": len(res["per_scenario"]),
        "N_fixes": res["ms_per_fix"]["n"],
        "airgap": {"socket_patched_to_raise": True, "probe_connection_blocked": blocked,
                   "network_modules_bound_in_veris.planner": net_mods},
        "notes": ["ground-truth mode: outdoors outdoor_nav; indoors moving indoor_nav; indoors still describe",
                  "planner runs at 1 Hz on every fix; plan(None) on GNSS dropout seconds",
                  "cloud_rules = context_engine.composer.Composer(use_llm=False) on the same local facts",
                  "cloud_rules indoor = situation is a venue type (the arbiter's own rule)"],
        **res,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / ("planner_dev.json" if a.dev else "planner.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("split", "N_fixes", "airgap", "overall_tick_weighted", "ms_per_fix")}, indent=1))
    print("->", path)


if __name__ == "__main__":
    sys.exit(main())
