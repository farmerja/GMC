"""
Mix files of several generate_size_sweep.py runs into one directory that ml/size_stream.py can train on: symlinks
part_XXXX.npz (and _subcell.npz when present), interleaved round-robin over the sources, plus a manifest.json whose
"settings" are the first source's (size_stream reads E_high from it) with the sources listed under "mix_of".

    python ml/mix_dirs.py --out ../ElectronTransport/mix_water_sx3 \
        --src ../ElectronTransport/sweep_water_3-10mm_5-30MeV_fwd_43:0-9 ../ElectronTransport/sweep_water_100um-1cm_30MeV_42:0,4,8,12,16,20,24,28,32,36
"""
import argparse
import json
import os


def parse(spec):
    d, _, sel = spec.partition(":")
    files = []
    for part in sel.split(","):
        a, _, b = part.partition("-")
        files += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return os.path.abspath(d), files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="+", required=True, help="<dir>:<file list>, e.g. dir:0-9 or dir:0,4,8")
    a = ap.parse_args()
    srcs = [parse(s) for s in a.src]
    mans = [json.load(open(os.path.join(d, "manifest.json"))) for d, _ in srcs]
    for (d, files), man in zip(srcs, mans):
        missing = [f for f in files if str(f) not in man["files"]]
        if missing:
            raise FileNotFoundError(f"{d}: files {missing} are not finished")
    order = []
    for i in range(max(len(f) for _, f in srcs)):
        for (d, files), man in zip(srcs, mans):
            if i < len(files):
                order.append((d, files[i], man))
    os.makedirs(a.out, exist_ok=True)
    out_man = {"settings": dict(mans[0]["settings"]), "mix_of": a.src, "files": {}}
    out_man["settings"]["E_high"] = max(float(m["settings"]["E_high"]) for m in mans)
    for k, (d, f, man) in enumerate(order):
        for suf in (".npz", "_subcell.npz"):
            src = os.path.join(d, f"part_{f:04d}{suf}")
            dst = os.path.join(a.out, f"part_{k:04d}{suf}")
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.exists(src):
                os.symlink(src, dst)
        out_man["files"][str(k)] = {**man["files"][str(f)], "source": f"{d}:{f}"}
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(out_man, fh, indent=1)
    print(f"{a.out}: {len(order)} files from {len(srcs)} source(s)")


if __name__ == "__main__":
    main()
