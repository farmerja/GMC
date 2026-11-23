#!/usr/bin/env python3
"""
Streaming, resumable training of the size-conditioned GMC and SCR kernel on generate_size_sweep.py data.

The data are a directory of self-contained files (ElectronTransport/generate_size_sweep.py, ~500k primaries each,
cell side L log-uniform in 100 um - 1 cm). Nothing ever holds more than one file: training VISITS the files one at a
time, trains every model on that file's rows, and saves the full training state (weights, optimizers, EMA, best
snapshots, history, position) after each visit. A run can therefore be stopped between files and resumed with the
same command:
    * --max_visits k   stop after k visits in this invocation;
    * touch <out>/STOP  stop after the current visit (the file is removed when the run stops);
    * killing the process loses at most the visit in progress.
--wait lets training start while the data are still being generated: a visit to a file that is not finished yet
polls the data directory's manifest until it is.

Plan: --sweeps passes over the --n_files files. The first sweep visits them in order (so training can follow the
generator); later sweeps in a seeded random order. Each model takes passes_total / sweeps passes over a file's rows per
visit, and its learning rate follows one warm-up + cosine schedule over the whole plan (by progress, so files of
different sizes need no step count up front). The files are independent samples of one distribution, so file-by-file
training is close to fully shuffled training.

Frozen at the start (from the first --spec_files files): the spec (energy range, log-L bounds, CSDA table), the
scattering-width table sigma(E, L) of --exit_scaled, the flow target standardization and the classifier prior.

    gmc  secondary_gmc models (clf_B, clf_I, exit_B, exit_I, daughter, photon[, stop]); exports
         <out>/secondary_gmc.pth (best-validation snapshot of each model, classifier temperatures fitted on the
         validation file) and secondary_gmc_last.pth after every visit, loadable by secondary_gmc.SecondaryGMC.
    scr  sub-cell kernel (subcell_kernel.SubcellFlow); exports <out>/subcell_kernel.pth (best EMA) and
         subcell_kernel_last.pth. --seg_weight size (default) weights every deposit by E / (mean transit dose at
         that cell size), so every log-L interval carries the same total weight; with plain dose weighting 1 cm cells
         (100x the dose per transit of 100 um cells) would take almost all of the capacity.

Usage (GMC venv, from the GMC root):
    python ml/size_stream.py gmc --data_dir ../ElectronTransport/sweep_water_100um-1cm_30MeV_42 --n_files 40 \
        --val_npz ../ElectronTransport/sweep_water_100um-1cm_30MeV_7/part_0000.npz \
        --out ml/models/size_gmc_water_30MeV --wait
    python ml/size_stream.py scr ... --out ml/models/size_scr_water_30MeV
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ml import secondary_gmc as sg  # noqa: E402
from ml import subcell_kernel as sk  # noqa: E402
from ml.exit_classifier import ExitClassifier, ExitClassifierConfig, fit_temperature  # noqa: E402
from ml.model import get_flow_matching_path, get_model  # noqa: E402


# ====================================================================================================================
# Files, plan, state
# ====================================================================================================================
def read_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        return json.load(f)


def file_path(data_dir, k):
    return os.path.join(data_dir, f"part_{k:04d}.npz")


def wait_for_file(data_dir, k, wait, poll=60):
    """Path of finished file k; with wait, poll the manifest until the generator has written it."""
    p = file_path(data_dir, k)
    announced = False
    while True:
        man = read_manifest(data_dir)
        if str(k) in man["files"] and os.path.exists(p) and os.path.exists(p[:-4] + "_subcell.npz"):
            return p
        if not wait:
            raise FileNotFoundError(f"{p} is not finished (pass --wait to wait for the generator)")
        if not announced:
            print(f"  waiting for {p} ...", flush=True)
            announced = True
        time.sleep(poll)


def visit_order(n_files, sweeps, seed):
    """Global visit list: sweep 0 in file order, later sweeps permuted (seeded)."""
    order = list(range(n_files))
    for s in range(1, sweeps):
        order += np.random.default_rng([seed, s]).permutation(n_files).tolist()
    return order


def lr_at(p, lr, warm):
    """One-cycle-like schedule by progress p in [0, 1]: linear warm-up from lr / 25, then cosine to ~0."""
    if p < warm:
        return lr * (0.04 + 0.96 * p / warm)
    q = min(1.0, (p - warm) / max(1e-9, 1.0 - warm))
    return lr * (1e-4 + (1 - 1e-4) * 0.5 * (1.0 + math.cos(math.pi * q)))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_cpu(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def should_stop(out, visits_this_run, max_visits):
    if os.path.exists(os.path.join(out, "STOP")):
        os.remove(os.path.join(out, "STOP"))
        print("STOP file found: stopping after this visit", flush=True)
        return True
    return max_visits is not None and visits_this_run >= max_visits


def check_plan(state, args, keys):
    """A resumed run must keep the plan (it defines the schedule): refuse silent changes."""
    old = state["args"]
    bad = {k: (old.get(k), getattr(args, k)) for k in keys if old.get(k) != getattr(args, k)}
    if bad:
        raise ValueError(f"resume with a different plan: {bad} (use a new --out, or match the old values)")


# ====================================================================================================================
# GMC
# ====================================================================================================================
# name -> (kind, builder(T) -> (cond, target/labels), passes attr, batch, warm-up, arch)
def gmc_model_table(args):
    t = {
        "clf_B": ("classifier", lambda T: sg.classifier_data(T, "boundary"), "clf_passes", 16384, 0.1),
        "clf_I": ("classifier", lambda T: sg.classifier_data(T, "interior"), "clf_passes", 16384, 0.1),
        "exit_B": ("flow", lambda T: sg.exit_flow_data(T, "boundary", scaled=args.exit_scaled)[:2], "flow_passes",
                   8192, 0.05),
        "exit_I": ("flow", lambda T: sg.exit_flow_data(T, "interior")[:2], "flow_passes", 8192, 0.05),
        "daughter": ("flow", sg.daughter_flow_data, "small_passes", 4096, 0.05),
        "photon": ("flow", sg.photon_flow_data, "small_passes", 4096, 0.05),
    }
    if args.stop:
        t["stop"] = ("flow", sg.stop_flow_data, "flow_passes", 8192, 0.05)
    if getattr(args, "train_only", None):
        t = {k: v for k, v in t.items() if k in args.train_only.split(",")}
    return t


def exit_sigma_rows(T):
    """(log10 E0, log10 L, v1 - v0) of the rows sigma(E, L) is measured on (as scatter_width_table)."""
    m = T["boundary"] & (T["cls"] == 1) & (np.abs(T["y0"] - 0.5) < 0.25)
    return np.log10(np.maximum(T["E0"][m], 1e-300)), np.log10(T["L"][m]), T["v1"][m] - T["v0"][m]


def load_frozen(args, dev):
    """Models of --init_from that are not retrained (inference format, kept as they are)."""
    _, loaded = sg.load_models(args.init_from, dev)
    keep = set(loaded) - set(gmc_model_table(args))
    return {k: {**loaded[k], "frozen": True} for k in keep}


def gmc_arch(name, args):
    if name.startswith("exit_"):
        return dict(hidden=args.exit_hidden, depth=args.exit_depth, expansion=2)
    if name == "photon":
        return dict(hidden=128, depth=4, expansion=2)
    return dict(hidden=256, depth=6, expansion=2)


def load_T(path, spec, NR):
    T = sg.load_tables(path, NR=NR)
    T["spec"] = spec
    return T


def merged_scatter_T(Ts, spec):
    keys = ("boundary", "cls", "y0", "E0", "v0", "v1", "L")
    return {"spec": spec, **{k: np.concatenate([T[k] for T in Ts]) for k in keys}}


def gmc_init(args, dev):
    man = read_manifest(args.data_dir)
    paths = [wait_for_file(args.data_dir, k, args.wait) for k in range(min(args.spec_files, args.n_files))]
    if args.init_from:
        spec, _ = sg.load_models(args.init_from, "cpu")
        print(f"  spec and frozen models from {args.init_from}; training {sorted(gmc_model_table(args))}")
        T0 = load_T(paths[0], spec, args.NR)
    else:
        T0 = sg.load_tables(paths[0], NR=args.NR)
        spec = T0["spec"]
        if not getattr(spec, "size_cond", False):
            raise ValueError(f"{paths[0]}: not a generate_size_sweep.py file (no seg_L)")
        spec.logE_hi = math.log10(float(man["settings"]["E_high"])) + 1e-9
        spec.edge_feats, spec.scale_feats = bool(args.edge_feats), False
    if args.smooth_exit_sigma:
        rows = [exit_sigma_rows(T0)] + [exit_sigma_rows(load_T(p, spec, args.NR)) for p in paths[1:]]
        lg, ll, dv = (np.concatenate(v) for v in zip(*rows))
        spec.sigx_logE, spec.sigx_logL, val = (v.tolist() for v in sg.smooth_scatter_table(lg, ll, dv, spec))
        spec.sigx_val = val
        print(f"  smooth exit sigma(E, L) from {len(paths)} file(s), {len(lg)} rows: 20 MeV at L = 1 / 2 / 4 mm "
              f"{np.round(sg.scatter_width(spec, np.full(3, 20.0), np.array([0.1, 0.2, 0.4]), 'exit'), 4)}")
    if not args.init_from and args.exit_scaled:
        Ts = [T0] + [load_T(p, spec, args.NR) for p in paths[1:]]
        sg.set_scatter_table(spec, merged_scatter_T(Ts, spec))
        del Ts
    T0["spec"] = spec
    models = {}
    for name, (kind, build, *_rest) in gmc_model_table(args).items():
        cond, tgt = build(T0)
        if kind == "classifier":
            cfg = ExitClassifierConfig(cond.shape[1], spec.n_classes(), 256, 4, 2, 0.0)
            m = ExitClassifier(cfg).to(dev)
            freq = np.bincount(tgt, minlength=cfg.num_classes) / len(tgt)
            with torch.no_grad():
                m.out[-1].bias.copy_(torch.as_tensor(np.log(np.maximum(freq, 0.1 / len(tgt))), dtype=torch.float32))
            models[name] = {"kind": kind, "model": m, "cfg": vars(cfg)}
            print(f"  {name}: cond {cond.shape[1]}, {spec.n_classes()} classes ({len(tgt)} rows in file 0)")
        else:
            a = gmc_arch(name, args)
            m = get_model(cond.shape[1] + tgt.shape[1], cond.shape[1], hidden_dim=a["hidden"], depth=a["depth"],
                          expansion_factor=a["expansion"]).to(dev)
            models[name] = {"kind": kind, "model": m, "arch": a, "mu": tgt.mean(0), "sd": tgt.std(0) + 1e-6,
                            "cond_dim": cond.shape[1], "target_dim": tgt.shape[1],
                            "scaled": bool(name == "exit_B" and args.exit_scaled), "rel": False}
            print(f"  {name}: cond {cond.shape[1]}, target {tgt.shape[1]} ({len(tgt)} rows in file 0)")
    for m in models.values():
        lr = args.clf_lr if m["kind"] == "classifier" else args.flow_lr
        wd = 1e-4 if m["kind"] == "classifier" else 1e-5
        m["opt"] = torch.optim.AdamW(m["model"].parameters(), lr=lr, weight_decay=wd)
        m["best"], m["best_val"], m["history"] = to_cpu(m["model"].state_dict()), float("inf"), []
    return spec, models


def gmc_val_data(args, spec, models, dev, n_max=200000):
    """Validation tensors of every model from the held-out file (fixed subsample)."""
    T = load_T(args.val_npz, spec, args.NR)
    rng = np.random.default_rng(1234)
    out = {}
    for name, (kind, build, *_rest) in gmc_model_table(args).items():
        cond, tgt = build(T)
        idx = rng.permutation(len(cond))[:n_max]
        c = torch.as_tensor(cond[idx], dtype=torch.float32, device=dev)
        if kind == "classifier":
            out[name] = (c, torch.as_tensor(tgt[idx], dtype=torch.long, device=dev))
        else:
            z = (tgt[idx] - models[name]["mu"]) / models[name]["sd"]
            out[name] = torch.cat([c, torch.as_tensor(z, dtype=torch.float32, device=dev)], 1)
    return out


def flow_loss(model, path, xb, cd, td, gen=None):
    x0 = torch.cat([xb[:, :cd], torch.randn(len(xb), td, device=xb.device, generator=gen)], 1)
    t = torch.rand(len(xb), device=xb.device, generator=gen)
    ps = path.sample(t=t, x_0=x0, x_1=xb)
    return ((model(ps.x_t, ps.t) - ps.dx_t)[:, cd:] ** 2).mean()


@torch.no_grad()
def gmc_val(name, m, V, dev):
    m["model"].eval()
    if m["kind"] == "classifier":
        c, y = V
        tot = sum(F.cross_entropy(m["model"](c[i:i + 65536]), y[i:i + 65536], reduction="sum").item()
                  for i in range(0, len(y), 65536))
        v = tot / len(y)
    else:
        path = get_flow_matching_path(cond_dim=m["cond_dim"])
        gen = torch.Generator(device=dev).manual_seed(1234)        # same noise every evaluation
        tot = sum(flow_loss(m["model"], path, V[i:i + 65536], m["cond_dim"], m["target_dim"], gen).item()
                  * len(V[i:i + 65536]) for i in range(0, len(V), 65536))
        v = tot / len(V)
    m["model"].train()
    return v


def gmc_visit(T, models, args, dev, visit, n_visits):
    """Train every model on the rows of one file. Returns {name: (rows, steps, mean train loss)}."""
    info = {}
    sweeps = args.sweeps
    for name, (kind, build, passes_attr, batch, warm) in gmc_model_table(args).items():
        m = models[name]
        cond, tgt = build(T)
        n = len(cond)
        if n == 0:
            continue
        steps = max(1, math.ceil(getattr(args, passes_attr) / sweeps * n / batch))
        if kind == "classifier":
            X = torch.as_tensor(cond, dtype=torch.float32, device=dev)
            Y = torch.as_tensor(tgt, dtype=torch.long, device=dev)
            lr0 = args.clf_lr
        else:
            X = torch.as_tensor(np.concatenate([cond, (tgt - m["mu"]) / m["sd"]], 1), dtype=torch.float32,
                                device=dev)
            path = get_flow_matching_path(cond_dim=m["cond_dim"])
            lr0 = args.flow_lr
        model, opt = m["model"], m["opt"]
        model.train()
        perm = torch.randperm(n, device=dev)
        pos, run = 0, 0.0
        for i in range(steps):
            if pos + batch > n:
                perm, pos = torch.randperm(n, device=dev), 0
            b = perm[pos:pos + batch]
            pos += batch
            set_lr(opt, lr_at((visit + i / steps) / n_visits, lr0, warm))
            if kind == "classifier":
                loss = F.cross_entropy(model(X[b]), Y[b])
            else:
                loss = flow_loss(model, path, X[b], m["cond_dim"], m["target_dim"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
        info[name] = (n, steps, run / steps)
        del X
    return info


def gmc_export(out, spec, models, V, dev, which):
    """Inference checkpoint in secondary_gmc.save_models format: 'best' (best-validation state of every model) or
    'last'. Classifier temperatures are fitted on the validation rows (on a copy; training is untouched)."""
    ex = {}
    for name, m in models.items():
        if m.get("frozen"):
            ex[name] = m
            continue
        mod = copy.deepcopy(m["model"])
        if which == "best":
            mod.load_state_dict(m["best"])
        mod.eval()
        if m["kind"] == "classifier":
            fit_temperature(mod, *V[name])
            ex[name] = {"kind": "classifier", "model": mod}
        else:
            ex[name] = {"kind": "flow", "model": mod, "mu": m["mu"], "sd": m["sd"], "cond_dim": m["cond_dim"],
                        "target_dim": m["target_dim"], "arch": m["arch"], "history": m["history"],
                        "rel": m["rel"], "scaled": m["scaled"]}
    tmp = os.path.join(out, f".export_{which}")
    sg.save_models(tmp, spec, ex)
    name = "secondary_gmc.pth" if which == "best" else "secondary_gmc_last.pth"
    os.replace(os.path.join(tmp, "secondary_gmc.pth"), os.path.join(out, name))
    os.rmdir(tmp)


GMC_PLAN_KEYS = ("data_dir", "n_files", "sweeps", "clf_passes", "flow_passes", "small_passes", "NR", "edge_feats",
                 "exit_scaled", "stop", "exit_hidden", "exit_depth", "clf_lr", "flow_lr", "seed", "init_from",
                 "train_only", "smooth_exit_sigma")


def cmd_gmc(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "state.pth")
    torch.manual_seed(args.seed)
    if os.path.exists(st_path):
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        check_plan(st, args, GMC_PLAN_KEYS)
        spec = sg.Spec(**st["spec"])
        models = {}
        for name, b in st["models"].items():
            if b["kind"] == "classifier":
                mod = ExitClassifier(ExitClassifierConfig(**b["cfg"])).to(dev)
            else:
                a = b["arch"]
                mod = get_model(b["cond_dim"] + b["target_dim"], b["cond_dim"], hidden_dim=a["hidden"],
                                depth=a["depth"], expansion_factor=a["expansion"]).to(dev)
            mod.load_state_dict(b["state"])
            opt = torch.optim.AdamW(mod.parameters(), lr=1e-3)
            opt.load_state_dict(b["opt"])
            models[name] = {**{k: v for k, v in b.items() if k not in ("state", "opt")}, "model": mod, "opt": opt}
        visit, log = st["visit"], st["log"]
        torch.set_rng_state(st["torch_rng"])
        if dev == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"Resumed {args.out} at visit {visit}", flush=True)
    else:
        print(f"New run {args.out}: initializing from the first {args.spec_files} file(s)", flush=True)
        spec, models = gmc_init(args, dev)
        visit, log = 0, []
    trained = {k: m for k, m in models.items()}
    if args.init_from:
        models = {**load_frozen(args, dev), **trained}
    V = gmc_val_data(args, spec, trained, dev)
    order = visit_order(args.n_files, args.sweeps, args.seed)
    n_visits = len(order)
    done_this_run = 0
    t_run = time.time()
    while visit < n_visits:
        k = order[visit]
        t0 = time.time()
        path = wait_for_file(args.data_dir, k, args.wait)
        T = load_T(path, spec, args.NR)
        t_load = time.time() - t0
        info = gmc_visit(T, models, args, dev, visit, n_visits)
        del T
        vals = {}
        for name, m in trained.items():
            v = gmc_val(name, m, V[name], dev)
            vals[name] = v
            m["history"].append((info.get(name, (0, 0, float("nan")))[2], v))
            if v < m["best_val"]:
                m["best_val"], m["best"] = v, to_cpu(m["model"].state_dict())
        visit += 1
        done_this_run += 1
        log.append({"visit": visit, "file": k, "load_s": round(t_load, 1), "s": round(time.time() - t0, 1),
                    "rows": {n: i[0] for n, i in info.items()}, "val": vals})
        print(f"visit {visit}/{n_visits} (sweep {(visit - 1) // args.n_files + 1}, file {k}): "
              f"load {t_load:.0f}s, total {time.time() - t0:.0f}s | val " +
              " ".join(f"{n} {v:.4f}" for n, v in vals.items()), flush=True)
        st = {"args": vars(args), "spec": vars(spec), "visit": visit, "log": log, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None, "models": {}}
        for name, m in trained.items():
            st["models"][name] = {**{k_: v for k_, v in m.items() if k_ not in ("model", "opt")},
                                  "state": m["model"].state_dict(), "opt": m["opt"].state_dict()}
        save_atomic(st, st_path)
        gmc_export(args.out, spec, models, V, dev, "best")
        gmc_export(args.out, spec, models, V, dev, "last")
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"spec": vars(spec), "args": vars(args), "log": log}, f, indent=1)
        if visit < n_visits and should_stop(args.out, done_this_run, args.max_visits):
            break
    print(f"{'Finished' if visit >= n_visits else 'Paused'} at visit {visit}/{n_visits} "
          f"({time.time() - t_run:.0f} s this run); exported {args.out}/secondary_gmc.pth", flush=True)


# ====================================================================================================================
# SCR kernel
# ====================================================================================================================
def scr_weights(R, ent, mode, n_L=20):
    """Per-entry sampling weight. 'dose': E; 'unit': E / transit dose; 'size': E / mean transit dose in the entry's
    log-L bin (equal total weight per log-L interval, dose weighting within it)."""
    w = ent["E"].astype(np.float64)
    if mode == "dose":
        return w
    if mode == "unit":
        return w / R["dose"][ent["seg"]]
    ll = np.log10(R["L"].astype(np.float64))
    edges = np.linspace(ll.min(), ll.max() + 1e-9, n_L + 1)
    b = np.clip(np.digitize(ll, edges) - 1, 0, n_L - 1)
    mean = np.bincount(b, weights=R["dose"], minlength=n_L) / np.maximum(np.bincount(b, minlength=n_L), 1)
    return w / mean[b][ent["seg"]]


def scr_tensors(R, ent, dev):
    Rd = sk.to_device(R, dev)
    return (Rd, torch.as_tensor(ent["seg"], device=dev), torch.as_tensor(ent["bin"], device=dev),
            torch.as_tensor(ent["mom"], device=dev), torch.as_tensor(ent["n"], device=dev))


SCR_PLAN_KEYS = ("data_dir", "n_files", "sweeps", "steps", "batch", "lr", "layers", "bins", "hidden", "ctx", "depth",
                 "n_freq", "seg_weight", "min_spread", "min_spread_cell", "seed")


def cmd_scr(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    st_path = os.path.join(args.out, "state.pth")
    man = read_manifest(args.data_dir)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    # validation: fixed weighted draw of entries of the held-out file, fixed points, no augmentation
    spec_v, Rv, ev = sk.load_kernel_data(args.val_npz)
    if st_exists := os.path.exists(st_path):
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        check_plan(st, args, SCR_PLAN_KEYS)
        spec = sk.Spec(**st["spec"])
    else:
        spec = spec_v
        spec.logE_hi = math.log10(float(man["settings"]["E_high"])) + 1e-9
        spec.dequant, spec.min_spread, spec.phase_n = "moment", args.min_spread, 0
        spec.min_spread_cell = args.min_spread_cell
    if not getattr(spec, "size_cond", False):
        raise ValueError("scr: needs generate_size_sweep.py data")
    rng = np.random.default_rng(1234)
    wv = scr_weights(Rv, ev, args.seg_weight)
    vidx = rng.choice(len(wv), size=min(args.n_val, len(wv)), p=wv / wv.sum())
    Rvd, vseg, vbin, vmom, vn = scr_tensors(Rv, {k: v[vidx] for k, v in ev.items()}, dev)
    torch.manual_seed(1234)
    vpts = sk.entry_points(spec, vbin, None, True, vmom, spec.min_spread, vn)
    del wv, ev
    model = sk.SubcellFlow(sk.cond_dim(spec), ctx_dim=args.ctx, hidden=args.hidden, n_layers=args.layers,
                           n_bins=args.bins, n_freq=args.n_freq, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ema = sk.EMA(model, args.ema)
    eval_model = sk.SubcellFlow(**model.arch).to(dev).eval()
    if st_exists:
        model.load_state_dict(st["state"])
        opt.load_state_dict(st["opt"])
        ema.shadow = {k: v.to(dev) for k, v in st["ema"].items()}
        visit, log, best = st["visit"], st["log"], st["best_val"]
        torch.set_rng_state(st["torch_rng"])
        if dev == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        gen.set_state(st["gen"])
        print(f"Resumed {args.out} at visit {visit}", flush=True)
    else:
        visit, log, best = 0, [], float("inf")
        torch.manual_seed(args.seed)
    print(f"Kernel: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, cond {sk.cond_dim(spec)}; "
          f"weights '{args.seg_weight}'", flush=True)

    @torch.no_grad()
    def val_nll():
        eval_model.load_state_dict(ema.shadow)
        tot = 0.0
        for i in range(0, len(vseg), 65536):
            sl = slice(i, i + 65536)
            idx = torch.arange(i, min(i + 65536, len(vseg)), device=dev)
            tot += sk.batch_nll(eval_model, spec, Rvd, vseg, vbin, idx, augment=False,
                                points=(vpts[0][sl], vpts[1][sl])).sum().item()
        return tot / len(vseg)

    order = visit_order(args.n_files, args.sweeps, args.seed)
    n_visits = len(order)
    done_this_run, t_run = 0, time.time()
    while visit < n_visits:
        k = order[visit]
        t0 = time.time()
        path = wait_for_file(args.data_dir, k, args.wait)
        _, R, ent = sk.load_kernel_data(path)
        w = scr_weights(R, ent, args.seg_weight)
        Rd, eseg, ebin, emom, en = scr_tensors(R, ent, dev)
        cdf = torch.as_tensor(np.cumsum(w), device=dev)
        total = cdf[-1].item()
        del R, ent, w
        t_load = time.time() - t0
        steps = max(1, round(args.steps / n_visits))
        run, nrun, n_skip = 0.0, 0, 0
        for i in range(steps):
            set_lr(opt, lr_at((visit + i / steps) / n_visits, args.lr, 0.03))
            u = torch.rand(args.batch, device=dev, dtype=torch.float64, generator=gen) * total
            idx = torch.searchsorted(cdf, u).clamp_max(len(cdf) - 1)
            loss = sk.batch_nll(model, spec, Rd, eseg, ebin, idx, gen, ent_mom=emom, ent_n=en).mean()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                if n_skip > 50:
                    raise RuntimeError("more than 50 non-finite training losses in one visit: diverged")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ema.update(model)
            run += loss.item()
            nrun += 1
        del Rd, eseg, ebin, emom, en, cdf
        vl = val_nll()
        visit += 1
        done_this_run += 1
        log.append({"visit": visit, "file": k, "steps": steps, "train": run / max(1, nrun), "val": vl,
                    "load_s": round(t_load, 1), "s": round(time.time() - t0, 1), "skipped": n_skip})
        print(f"visit {visit}/{n_visits} (file {k}): {steps} steps, train NLL {run / max(1, nrun):.4f}, "
              f"val NLL (EMA) {vl:.4f} | load {t_load:.0f}s, total {time.time() - t0:.0f}s"
              + (f" [{n_skip} non-finite skipped]" if n_skip else ""), flush=True)
        if vl < best:
            best = vl
            sk.save_kernel(os.path.join(args.out, "subcell_kernel.pth"), spec, eval_model, log, vars(args))
        sk.save_kernel(os.path.join(args.out, "subcell_kernel_last.pth"), spec, eval_model, log, vars(args))
        save_atomic({"args": vars(args), "spec": vars(spec), "visit": visit, "log": log, "best_val": best,
                     "state": model.state_dict(), "opt": opt.state_dict(), "ema": ema.shadow,
                     "torch_rng": torch.get_rng_state(), "gen": gen.get_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None}, st_path)
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"spec": vars(spec), "args": vars(args), "best_val_nll": best, "log": log}, f, indent=1)
        if visit < n_visits and should_stop(args.out, done_this_run, args.max_visits):
            break
    print(f"{'Finished' if visit >= n_visits else 'Paused'} at visit {visit}/{n_visits} "
          f"({time.time() - t_run:.0f} s this run); best val NLL {best:.4f}", flush=True)


# ====================================================================================================================
# CLI
# ====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gmc", "scr"):
        p = sub.add_parser(name)
        p.add_argument("--data_dir", required=True, help="generate_size_sweep.py output directory")
        p.add_argument("--n_files", type=int, required=True, help="files in the plan (fixed for the run)")
        p.add_argument("--val_npz", required=True, help="held-out file (another seed's part_0000.npz)")
        p.add_argument("--out", required=True)
        p.add_argument("--wait", action="store_true", help="wait for files the generator has not finished yet")
        p.add_argument("--max_visits", type=int, default=None, help="stop after this many visits in this invocation")
        p.add_argument("--seed", type=int, default=0)
    g = sub.choices["gmc"]
    g.add_argument("--sweeps", type=int, default=6, help="passes over the file list")
    g.add_argument("--clf_passes", type=float, default=15.0, help="total passes over the data, classifiers")
    g.add_argument("--flow_passes", type=float, default=60.0, help="total passes, exit (and stop) flows")
    g.add_argument("--small_passes", type=float, default=60.0, help="total passes, daughter / photon flows")
    g.add_argument("--clf_lr", type=float, default=2e-3)
    g.add_argument("--flow_lr", type=float, default=1e-3)
    g.add_argument("--NR", type=int, default=3)
    g.add_argument("--exit_hidden", type=int, default=256)
    g.add_argument("--exit_depth", type=int, default=6)
    g.add_argument("--no_edge_feats", dest="edge_feats", action="store_false")
    g.add_argument("--no_exit_scaled", dest="exit_scaled", action="store_false")
    g.add_argument("--stop", action="store_true", help="also train the stop-point flow (charge tally)")
    g.add_argument("--spec_files", type=int, default=4, help="files used to measure sigma(E, L) at the start")
    g.add_argument("--init_from", default=None,
                   help="existing secondary_gmc.pth: its spec and every model not in --train_only are kept as they are")
    g.add_argument("--train_only", default=None, help="comma-separated models to (re)train from scratch, e.g. exit_B")
    g.add_argument("--smooth_exit_sigma", action="store_true",
                   help="exit-flow scale from a smooth sigma(E, L) over the --spec_files files (spec.sigx_*)")
    s = sub.choices["scr"]
    s.add_argument("--sweeps", type=int, default=3)
    s.add_argument("--steps", type=int, default=160000, help="optimizer steps over the whole plan")
    s.add_argument("--batch", type=int, default=16384)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--clip", type=float, default=1.0)
    s.add_argument("--ema", type=float, default=0.999)
    s.add_argument("--layers", type=int, default=8)
    s.add_argument("--bins", type=int, default=32)
    s.add_argument("--hidden", type=int, default=256)
    s.add_argument("--ctx", type=int, default=256)
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--n_freq", type=int, default=6)
    s.add_argument("--seg_weight", choices=("size", "dose", "unit"), default="size")
    s.add_argument("--min_spread", type=float, default=0.05, help="dequantization spread floor (bin widths)")
    s.add_argument("--min_spread_cell", type=float, default=5e-4,
                   help="... and (cell widths): 5e-4 = half a bin at 1000 x 1000 (0 = off, the first run)")
    s.add_argument("--n_val", type=int, default=500000)
    args = ap.parse_args()
    (cmd_gmc if args.cmd == "gmc" else cmd_scr)(args)


if __name__ == "__main__":
    main()
