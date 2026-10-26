"""Replay control studies P620-P626 (CPU, zero API).

Consolidates scripts/p6xx_common.py and run_p620..run_p626. Every study runs
on the verified main-table BM25 replay (SparseStaticIndex per conversation,
NativeSuppression strength .5 kappa 3 pool 100, released question order) and
must reproduce the published evidence Recall@10 (off 51.94 / count 53.41 on
the 1,527 evidence-valid LoCoMo questions) before any cell is scored.

Subcommands: injection (P620), sufficiency (P621), exposure (P622),
evolvability (P623), ceilings (P624), swap (P625), granularity (P626).

P622 reads per-question arm deltas from results/locomo_arm_deltas.json,
exported by `python -m src.runs.score_answers retrieval` (the clean-repo
replacement for the legacy p582 mechanism record), and asserts the replayed
count-arm mean matches the saved deltas to 1e-6.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict

from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.runs.common import (RECALL_BOOT_DRAWS, RECALL_BOOT_SEED, RESULTS,
                             cluster_boot, locomo_world, recall)

PUBLISHED = dict(off=51.94, count=53.41)


def gate(world):
    """Reproduce the published off/count arms; returns per-question recalls."""
    off, cnt = {}, {}
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
        for q in world["by_conv"][conv]:
            native = [u for u, _ in host.search(q.question, 10)]
            ids = [u for u, _ in op.search(q.question, 10)]
            op.observe(native, [1] * len(native))
            if q.qid in world["gold"]:
                gold = world["gold"][q.qid]
                off[q.qid] = recall(native, gold)
                cnt[q.qid] = recall(ids, gold)
    o = round(100 * statistics.mean(off.values()), 2)
    c = round(100 * statistics.mean(cnt.values()), 2)
    assert (o, c) == (PUBLISHED["off"], PUBLISHED["count"]), (o, c)
    return off, cnt


def write(name, payload):
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", path)


LEVELS = (0, 1, 2, 4, 8, 16, 32)
TARGETS = ("competitors", "gold", "below_cut", "random_pool")


def cmd_injection(args):
    """P620: causal traffic injection; only competitor access history changes."""
    plan = dict(study="p620_traffic_injection", levels=LEVELS, targets=TARGETS,
                protocol="main-table BM25 replay; fresh state per question; kappa 3 strength .5 pool 100",
                sealed_before_scoring=True, boot=dict(draws=RECALL_BOOT_DRAWS, seed=RECALL_BOOT_SEED))
    write("p620_traffic_injection_plan.json", plan)
    world = locomo_world()
    off_pq, cnt_pq = gate(world)
    gold = world["gold"]
    rng = random.Random(RECALL_BOOT_SEED)
    per = {t: {n: {} for n in LEVELS} for t in TARGETS}
    natural = defaultdict(list)
    t0 = time.time()
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        online = NativeSuppression(host, strength=.5, kappa=3, pool=100)
        for q in world["by_conv"][conv]:
            native = [u for u, _ in host.search(q.question, 10)]
            if q.qid in gold:
                G = gold[q.qid]
                pool = [u for u, _ in host.search(q.question, 100)]
                comp = [u for u in native if u not in G]
                gold_top = [u for u in native if u in G]
                below = [u for u in pool[10:] if u not in G][:10]
                cand = [u for u in pool[10:] if u not in G]
                rand = rng.sample(cand, min(10, len(cand)))
                natural["competitor_n"].extend(online.counts[u] for u in comp)
                natural["gold_n"].extend(online.counts[u] for u in gold_top)
                for tname, tset in (("competitors", comp), ("gold", gold_top),
                                    ("below_cut", below), ("random_pool", rand)):
                    for n in LEVELS:
                        op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
                        for u in tset:
                            op.counts[u] = n
                        ids = [u for u, _ in op.search(q.question, 10)]
                        per[tname][n][q.qid] = recall(ids, G)
            online.observe(native, [1] * len(native))
    out = dict(status="complete_zero_call", plan=plan,
               verification=dict(off=PUBLISHED["off"], count=PUBLISHED["count"], reproduced=True),
               seconds=round(time.time() - t0, 1), results={}, natural={})
    for tname in TARGETS:
        rows = {}
        for n in LEVELS:
            delta = {qid: per[tname][n][qid] - per[tname][0][qid] for qid in per[tname][n]}
            rows[n] = dict(recall=round(100 * statistics.mean(per[tname][n].values()), 3),
                           delta=cluster_boot(delta),
                           improved=sum(1 for v in delta.values() if v > 0),
                           worsened=sum(1 for v in delta.values() if v < 0))
        out["results"][tname] = rows
    for k, vs in natural.items():
        out["natural"][k] = dict(n=len(vs), mean=round(statistics.mean(vs), 3),
                                 median=statistics.median(vs),
                                 share_ge={str(n): round(sum(1 for v in vs if v >= n) / len(vs), 3)
                                           for n in LEVELS[1:]})
    out["online_count_delta"] = cluster_boot({q: cnt_pq[q] - off_pq[q] for q in off_pq})
    write("p620_traffic_injection.json", out)
    for tname in TARGETS:
        print(tname, {n: out["results"][tname][n]["delta"]["mean"] for n in LEVELS})


FRACS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)


def cmd_sufficiency(args):
    """P621: how much observed traffic before the index is worth using."""
    plan = dict(study="p621_traffic_sufficiency", fractions=FRACS,
                split="first 75% accumulate / last 25% held-out future set",
                protocol="main-table BM25 replay; kappa 3 strength .5 pool 100; released order",
                sealed_before_scoring=True)
    write("p621_traffic_sufficiency_plan.json", plan)
    world = locomo_world()
    gate(world)
    gold = world["gold"]
    per = {f: {} for f in FRACS}
    online, off = {}, {}
    seen_q = {f: [] for f in FRACS}
    t0 = time.time()
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        qs = world["by_conv"][conv]
        cut = int(round(0.75 * len(qs)))
        acc, future = qs[:cut], qs[cut:]
        for f in FRACS:
            k = int(round(f * len(acc)))
            op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
            for q in acc[:k]:
                native = [u for u, _ in host.search(q.question, 10)]
                op.observe(native, [1] * len(native))
            seen_q[f].append(k)
            for q in future:
                if q.qid in gold:
                    ids = [u for u, _ in op.search(q.question, 10)]
                    per[f][q.qid] = recall(ids, gold[q.qid])
        op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
        for q in qs:
            native = [u for u, _ in host.search(q.question, 10)]
            ids = [u for u, _ in op.search(q.question, 10)]
            op.observe(native, [1] * len(native))
            if q in future and q.qid in gold:
                online[q.qid] = recall(ids, gold[q.qid])
                off[q.qid] = recall(native, gold[q.qid])
    out = dict(status="complete_zero_call", plan=plan, seconds=round(time.time() - t0, 1),
               future_set=dict(n_questions=len(off),
                               off_recall=round(100 * statistics.mean(off.values()), 3)),
               frozen={}, online=cluster_boot({q: online[q] - off[q] for q in off}))
    for f in FRACS:
        d = {q: per[f][q] - off[q] for q in off}
        out["frozen"][str(f)] = dict(questions_observed=dict(total=sum(seen_q[f]), per_conv=seen_q[f]),
                                     recall=round(100 * statistics.mean(per[f].values()), 3),
                                     delta=cluster_boot(d))
    write("p621_traffic_sufficiency.json", out)
    print("future n", len(off), "off", out["future_set"]["off_recall"], "online", out["online"]["mean"])


EK = (("0-1", 0, 1), ("2-3", 2, 3), ("4+", 4, 99))
GX = (("0", 0, 0), ("1-2", 1, 2), ("3-9", 3, 9), ("10+", 10, 10**9))


def bucket(v, spec):
    for name, lo, hi in spec:
        if lo <= v <= hi:
            return name


def cmd_exposure(args):
    """P622: competitor exposure x gold exposure, count against semantic arms."""
    plan = dict(study="p622_exposure_map", ek_buckets=[e[0] for e in EK],
                gold_buckets=[g[0] for g in GX],
                source="results/locomo_arm_deltas.json per-question recall deltas (bm25 count/frozen/trained)",
                sealed_before_scoring=True)
    write("p622_exposure_map_plan.json", plan)
    saved_path = RESULTS / "locomo_arm_deltas.json"
    saved_doc = json.loads(saved_path.read_text())
    assert saved_doc["status"] == "complete"
    saved_arms = saved_doc["arms"]
    for required in ("count", "frozen", "trained"):
        assert required in saved_arms, f"{saved_path} lacks the {required} arm"
    world = locomo_world()
    off, cnt = gate(world)
    gold = world["gold"]
    ek, gx = {}, {}
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        seen = defaultdict(int)
        for q in world["by_conv"][conv]:
            native = [u for u, _ in host.search(q.question, 10)]
            if q.qid in gold:
                ek[q.qid] = sum(1 for u in native if seen[u] > 0)
                gx[q.qid] = sum(seen[u] for u in gold[q.qid])
            for u in native:
                seen[u] += 1
    mine = {q: cnt[q] - off[q] for q in off}
    saved = {q: saved_arms["count"][q] for q in off}
    assert abs(statistics.mean(mine.values()) - statistics.mean(saved.values())) < 1e-6, \
        "saved count deltas do not match the replay"
    arms = {a: {q: saved_arms[a][q] for q in off} for a in ("count", "frozen", "trained")}
    cells = {}
    for ename, *_ in EK:
        for gname, *_ in GX:
            qs = [q for q in off if bucket(ek[q], EK) == ename and bucket(gx[q], GX) == gname]
            if not qs:
                continue
            cell = dict(n=len(qs))
            for a in arms:
                cell[a] = cluster_boot({q: arms[a][q] for q in qs})
            cell["count_minus_frozen"] = cluster_boot(
                {q: arms["count"][q] - arms["frozen"][q] for q in qs})
            cells[f"{ename}|{gname}"] = cell
    margins = dict(
        ek={e[0]: cluster_boot({q: arms["count"][q] - arms["frozen"][q]
                                for q in off if bucket(ek[q], EK) == e[0]}) for e in EK},
        gold={g[0]: cluster_boot({q: arms["count"][q] - arms["frozen"][q]
                                  for q in off if bucket(gx[q], GX) == g[0]}) for g in GX})
    out = dict(status="complete_zero_call", plan=plan, cells=cells,
               count_minus_frozen_margins=margins,
               overall={a: cluster_boot(arms[a]) for a in arms})
    write("p622_exposure_map.json", out)
    for k, c in cells.items():
        print("%-8s n=%4d count %+.2f frozen %+.2f trained %+.2f  c-f %+.2f [%+.2f,%+.2f]" % (
            k, c["n"], c["count"]["mean"], c["frozen"]["mean"], c["trained"]["mean"],
            c["count_minus_frozen"]["mean"], c["count_minus_frozen"]["lo"],
            c["count_minus_frozen"]["hi"]))


RO_BINS = (("low", 0.0, 0.4), ("mid", 0.4, 0.8), ("high", 0.8, 1.01))
HR_BANDS = (("11-20", 11, 20), ("21-50", 21, 50), ("51-100", 51, 100), (">100", 101, 10**9))


def band(v, spec):
    for name, lo, hi in spec:
        if lo <= v < hi if spec is RO_BINS else lo <= v <= hi:
            return name


def cmd_evolvability(args):
    """P623: index evolvability map, competition x headroom."""
    plan = dict(study="p623_evolvability_map", ro_bins=[b[0] for b in RO_BINS],
                headroom_bands=[b[0] for b in HR_BANDS], levels=LEVELS, sealed_before_scoring=True)
    write("p623_evolvability_map_plan.json", plan)
    world = locomo_world()
    off, cnt = gate(world)
    gold = world["gold"]
    ro, hclass, hband = {}, {}, {}
    inj = {n: {} for n in LEVELS}
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        seen = defaultdict(int)
        for q in world["by_conv"][conv]:
            native = [u for u, _ in host.search(q.question, 10)]
            if q.qid in gold:
                G = gold[q.qid]
                pool = [u for u, _ in host.search(q.question, 100)]
                ro[q.qid] = sum(1 for u in native if seen[u] > 0) / len(native)
                missing = [u for u in G if u not in native]
                ranks = [pool.index(u) + 1 for u in missing if u in pool]
                if not missing:
                    hclass[q.qid] = "none_needed"
                elif ranks:
                    hclass[q.qid] = "reachable"
                else:
                    hclass[q.qid] = "unreachable"
                hband[q.qid] = band(min(ranks), HR_BANDS) if ranks else (">100" if missing else "none")
                comp = [u for u in native if u not in G]
                for n in LEVELS:
                    op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
                    for u in comp:
                        op.counts[u] = n
                    inj[n][q.qid] = recall([u for u, _ in op.search(q.question, 10)], G)
            for u in native:
                seen[u] += 1
    delta = {q: cnt[q] - off[q] for q in off}
    obs = {}
    for rname, *_ in RO_BINS:
        for hc in ("none_needed", "reachable", "unreachable"):
            qs = [q for q in off if band(ro[q], RO_BINS) == rname and hclass[q] == hc]
            if qs:
                obs[f"{rname}|{hc}"] = dict(
                    n=len(qs), count_delta=cluster_boot({q: delta[q] for q in qs}),
                    off_recall=round(100 * statistics.mean(off[q] for q in qs), 2))
    ctrl = {}
    for hb, *_ in HR_BANDS:
        qs = [q for q in off if hband[q] == hb]
        if not qs:
            continue
        ctrl[hb] = dict(n=len(qs), levels={str(n): cluster_boot(
            {q: inj[n][q] - inj[0][q] for q in qs}) for n in LEVELS})
    out = dict(status="complete_zero_call", plan=plan,
               headroom_shares={k: sum(1 for q in off if hclass[q] == k) / len(off)
                                for k in ("none_needed", "reachable", "unreachable")},
               ro_summary=dict(mean=round(statistics.mean(ro.values()), 3),
                               bins={b[0]: sum(1 for q in off if band(ro[q], RO_BINS) == b[0])
                                     for b in RO_BINS}),
               observational=obs, controlled=ctrl)
    write("p623_evolvability_map.json", out)
    print("headroom shares", out["headroom_shares"], "RO", out["ro_summary"])


def cmd_ceilings(args):
    """P624: how much of the reachable index headroom the count rule captures."""
    plan = dict(study="p624_oracle_ceilings",
                arms=["count", "form_ceiling", "reorder_ceiling", "expansion_ceiling"],
                sealed_before_scoring=True)
    write("p624_oracle_ceilings_plan.json", plan)
    world = locomo_world()
    off, cnt = gate(world)
    gold = world["gold"]
    form, reorder, expansion = {}, {}, {}
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        for q in world["by_conv"][conv]:
            if q.qid not in gold:
                continue
            G = gold[q.qid]
            native = [u for u, _ in host.search(q.question, 10)]
            pool = [u for u, _ in host.search(q.question, 100)]
            op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
            for u in native:
                if u not in G:
                    op.counts[u] = 10**9
            form[q.qid] = recall([u for u, _ in op.search(q.question, 10)], G)
            in_pool = [u for u in pool if u in G]
            reorder[q.qid] = min(len(in_pool), 10) / len(G)
            expansion[q.qid] = 1.0
    arms = dict(count={q: cnt[q] - off[q] for q in off},
                form_ceiling={q: form[q] - off[q] for q in off},
                reorder_ceiling={q: reorder[q] - off[q] for q in off},
                expansion_ceiling={q: expansion[q] - off[q] for q in off})
    res = {a: cluster_boot(v) for a, v in arms.items()}
    c = res["count"]["mean"]
    out = dict(status="complete_zero_call", plan=plan, off_recall=PUBLISHED["off"], arms=res,
               fraction_captured={a: round(c / res[a]["mean"], 3) for a in arms if a != "count"},
               saved_hindsight_oracles=dict(eta_only=3.13, rho_eta=2.24,
                                            source="wave-2 replay (quoted, not recomputed)"),
               fraction_of_hindsight={k: round(c / v, 3) for k, v in dict(eta_only=3.13, rho_eta=2.24).items()})
    write("p624_oracle_ceilings.json", out)
    for a, r in res.items():
        print("%-18s %+.2f [%+.2f,%+.2f]" % (a, r["mean"], r["lo"], r["hi"]))


def _build(host, qs, k):
    op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
    for q in qs[:k]:
        native = [u for u, _ in host.search(q.question, 10)]
        op.observe(native, [1] * len(native))
    return op


def _score(op, future, gold):
    out = {}
    for q in future:
        if q.qid in gold:
            out[q.qid] = recall([u for u, _ in op.search(q.question, 10)], gold[q.qid])
    return out


def _run_split(world, streams_of):
    gold = world["gold"]
    res = defaultdict(dict)
    offs = {}
    for conv, units in world["groups"].items():
        host = SparseStaticIndex(units)
        streams = streams_of(world["by_conv"][conv])
        states, futures = {}, {}
        for name, qs in streams.items():
            cut = int(round(0.75 * len(qs)))
            states[name] = _build(host, qs, cut)
            futures[name] = qs[cut:]
        for name, fut in futures.items():
            empty = NativeSuppression(host, strength=.5, kappa=3, pool=100)
            offs.update(_score(empty, fut, gold))
            res[f"{name}|matched"].update(_score(states[name], fut, gold))
            union = NativeSuppression(host, strength=.5, kappa=3, pool=100)
            for other, st in states.items():
                if other == name:
                    continue
                res[f"{name}|swap:{other}"].update(_score(st, fut, gold))
                for u, c in st.counts.items():
                    union.counts[u] += c
            res[f"{name}|union_others"].update(_score(union, fut, gold))
    summary = {}
    for arm, per in res.items():
        d = {q: per[q] - offs[q] for q in per}
        summary[arm] = cluster_boot(d)
    paired = {}
    for q_arm, per in res.items():
        name, kind = q_arm.split("|", 1)
        if not kind.startswith("swap:"):
            continue
        for q, v in per.items():
            paired.setdefault(q, []).append(res[f"{name}|matched"][q] - v)
    summary["paired_matched_minus_swap"] = cluster_boot(
        {q: statistics.mean(v) for q, v in paired.items()})
    summary["paired_matched_minus_union"] = cluster_boot(
        {q: res[f'{a.split("|")[0]}|matched'][q] - per[q]
         for a, per in res.items() if a.endswith("|union_others") for q in per})
    return summary


def cmd_swap(args):
    """P625: counterfactual content-index swap and same-store transfer."""
    plan = dict(study="p625_index_swap", splits=["category", "interleaved"],
                accumulate="first 75% of each stream", future="last 25%",
                sealed_before_scoring=True)
    write("p625_index_swap_plan.json", plan)
    world = locomo_world()
    gate(world)
    cat = world["cat"]

    def by_category(qs):
        s = defaultdict(list)
        for q in qs:
            s[f"cat{cat[q.qid]}"].append(q)
        return {k: v for k, v in s.items() if len(v) >= 8}

    def interleaved(qs):
        return dict(even=qs[0::2], odd=qs[1::2])

    out = dict(status="complete_zero_call", plan=plan,
               category=_run_split(world, by_category),
               interleaved=_run_split(world, interleaved))
    write("p625_index_swap.json", out)
    for split in ("category", "interleaved"):
        for arm, r in sorted(out[split].items()):
            print("%-12s %-22s n=%4d %+.2f [%+.2f,%+.2f]" % (
                split, arm, r["n"], r["mean"], r["lo"], r["hi"]))


RUNGS = (1, 2, 4, 8, "session")


class Unit:
    def __init__(self, uid, conv_id, turns, text):
        self.uid, self.conv_id, self.turns, self._text = uid, conv_id, turns, text

    def content(self):
        return self._text


def _make_units(items, g):
    by_sess = defaultdict(list)
    for it in items:
        by_sess[(it.conv_id, it.session)].append(it)
    units, member = [], {}
    for (conv, sess), turns in by_sess.items():
        turns.sort(key=lambda t: int(t.turn_index))
        size = len(turns) if g == "session" else g
        for i in range(0, len(turns), size):
            chunk = turns[i:i + size]
            uid = f"{conv}::S{sess}:U{i // size}"
            units.append(Unit(uid, conv, [t.uid for t in chunk],
                              " ".join(t.content() for t in chunk)))
            for t in chunk:
                member[t.uid] = uid
    return units, member


def cmd_granularity(args):
    """P626: memory granularity against index evolvability."""
    plan = dict(study="p626_granularity", rungs=[str(r) for r in RUNGS], sealed_before_scoring=True)
    write("p626_granularity_plan.json", plan)
    world = locomo_world()
    gate(world)
    gold = world["gold"]
    out = dict(status="complete_zero_call", plan=plan, rungs={})
    for g in RUNGS:
        off, cnt, purity, mixing, nunits = {}, {}, [], [], 0
        for conv, items in world["groups"].items():
            units, member = _make_units(items, g)
            nunits += len(units)
            umap = {u.uid: u for u in units}
            host = SparseStaticIndex(units)
            op = NativeSuppression(host, strength=.5, kappa=3, pool=100)
            for q in world["by_conv"][conv]:
                native = [u for u, _ in host.search(q.question, 10)]
                ids = [u for u, _ in op.search(q.question, 10)]
                op.observe(native, [1] * len(native))
                if q.qid in gold:
                    G = {member[t] for t in gold[q.qid]}
                    off[q.qid] = recall(native, G)
                    cnt[q.qid] = recall(ids, G)
                    for u in G:
                        turns = umap[u].turns
                        purity.append(sum(1 for t in turns if t in gold[q.qid]) / len(turns))
                        mixing.append(len(turns))
        d = {q: cnt[q] - off[q] for q in off}
        out["rungs"][str(g)] = dict(
            units=nunits, off_recall=round(100 * statistics.mean(off.values()), 3),
            count_recall=round(100 * statistics.mean(cnt.values()), 3),
            delta=cluster_boot(d), purity=round(statistics.mean(purity), 4),
            turns_per_gold_unit=round(statistics.mean(mixing), 3),
            improved=sum(1 for v in d.values() if v > 0),
            worsened=sum(1 for v in d.values() if v < 0))
        r = out["rungs"][str(g)]
        print("g=%-8s units %6d off %.2f count %.2f delta %+.2f [%+.2f,%+.2f] purity %.3f turns/unit %.2f up %d down %d" % (
            g, nunits, r["off_recall"], r["count_recall"], r["delta"]["mean"], r["delta"]["lo"],
            r["delta"]["hi"], r["purity"], r["turns_per_gold_unit"], r["improved"], r["worsened"]))
    write("p626_granularity.json", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("injection", cmd_injection), ("sufficiency", cmd_sufficiency),
                     ("exposure", cmd_exposure), ("evolvability", cmd_evolvability),
                     ("ceilings", cmd_ceilings), ("swap", cmd_swap),
                     ("granularity", cmd_granularity)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
