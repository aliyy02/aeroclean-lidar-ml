"""Re-label a whole dataset with ONE global GT correction (the tape error is global) + the
user's scheme, and re-export bags.

Fit one `register_dataset` correction from a spread of scans, then label every frame of every
bag with it (floor is still found per-bag). Outputs per-frame npz + an mcap bag with
x,y,z,intensity,label.
"""
import sys
import os
import csv
import json
import numpy as np

sys.path.insert(0, ".")
from calibration.io_bag import read_all
from calibration.gt_parse import load_gt
from calibration.label import detect_floor
from calibration.frames import apply_roll
from analysis.seg import axis_map, plane_basis
from calibration import register as reg
from analysis.labeled_to_bag import write_scan_bag

CFG = {
    "oxy": ("Oxy_L2_All/Oxy_L2_test_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
            "Oxy_{n}", list(range(1, 60))),
    "bech": ("Bech_all/Bech_109_test_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt",
             "Bech_{n}", list(range(1, 41))),
}
NAMES = {0: "not_glass", 1: "glass", 2: "ground", 3: "interior"}


def agg_ned(bag):
    frames = read_all(bag)
    ned = np.vstack([apply_roll(axis_map(f.xyz.astype(float), 1), 24.0) for f in frames])
    inten = np.concatenate([f.intensity.astype(float) for f in frames])
    return frames, ned, inten


def main():
    ds = sys.argv[1]
    bagt, gtt, namet, ns = CFG[ds]
    present = [n for n in ns if os.path.isdir(bagt.format(n=n))]

    # --- fit ONE global correction from a spread of scans ---
    fit_ns = present[:: max(1, len(present) // 12)][:12]
    items = []
    for n in fit_ns:
        _, ned, inten = agg_ned(bagt.format(n=n))
        items.append((ned, inten, load_gt(gtt.format(n=n))))
    corr = reg.register_dataset(items)
    print(f"{ds}: GLOBAL correction from {len(items)} scans -> "
          f"mp={corr.mp:+.3f} dd={corr.dd:+.3f} du={corr.du:+.3f} dv={corr.dv:+.3f} "
          f"da={corr.da:+.1f} trim={corr.delta:.3f}")

    npz_root, bag_root = f"data/labeled_{ds}", f"data/labeled_bags/{ds}"
    import shutil
    for d in (npz_root, bag_root):
        if os.path.isdir(d):
            shutil.rmtree(d)                       # start clean (no stale frames)
    os.makedirs(npz_root, exist_ok=True)
    ok = 0
    for n in present:
        frames, ned_all, _ = agg_ned(bagt.format(n=n))
        gt = load_gt(gtt.format(n=n))
        floor = detect_floor(ned_all, gt)
        # pass 1: label + PERPENDICULAR-project every frame to de-rotated (uc,vc) [matches label()]
        nrm, e_u, e_v, O = plane_basis(gt)
        per = []
        for f in frames:
            ned_f = apply_roll(axis_map(f.xyz.astype(float), 1), 24.0)
            lab = reg.label(ned_f, f.intensity.astype(float), gt, corr, floor=floor)
            rel = ned_f - O
            uc, vc = reg._apply_uv(rel @ e_u, rel @ e_v, corr)
            behind_f = -(ned_f @ nrm + gt.plane_d)
            per.append((f.stamp_ns, ned_f, f.intensity.astype(float), lab, uc, vc, behind_f))
        # AGGREGATE region-grow: glass connected to frame -> frame; for Bech ALSO the near-plane
        # (recessed inner-frame) interior. (Oxy's near-plane curtain is continuous and would
        # over-convert, so interior-grow is Bech-only -- user: leave Oxy as-is.)
        off = np.cumsum([0] + [len(p[3]) for p in per])
        beh = np.concatenate([p[6] for p in per]) if ds == "bech" else None
        big = reg.reclassify_glass_at_frame(
            np.concatenate([p[3] for p in per]), np.concatenate([p[4] for p in per]),
            np.concatenate([p[5] for p in per]), behind=beh, dd=corr.dd)
        # pass 2: split back + save
        out = os.path.join(npz_root, namet.format(n=n)); os.makedirs(out, exist_ok=True)
        rows = []
        for i, (stamp, ned_f, inten_f, _lab, uc, vc, _b) in enumerate(per):
            lab = big[off[i]:off[i + 1]]
            keep = lab != reg.DROP
            fn = f"frame_{i:04d}.npz"
            np.savez_compressed(os.path.join(out, fn), xyz=ned_f[keep].astype(np.float32),
                                intensity=inten_f[keep].astype(np.float32),
                                label=lab[keep].astype(np.int16),
                                normal=np.tile(gt.plane_normal, (int(keep.sum()), 1)).astype(np.float32))
            counts = {nm: int((lab[keep] == k).sum()) for k, nm in NAMES.items()}
            rows.append({"file": fn, "stamp_ns": stamp, "n": int(keep.sum()), **counts})
        with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(os.path.join(out, "meta.json"), "w") as fh:
            json.dump({"correction": vars(corr), "alpha": 24.0, "global_fit": True}, fh, indent=2)
        write_scan_bag(out, os.path.join(bag_root, namet.format(n=n)))
        ok += 1
    # aggregate mix
    tot = {}
    for d in (os.path.join(npz_root, namet.format(n=n)) for n in present):
        for r in csv.DictReader(open(os.path.join(d, "index.csv"))):
            for k in NAMES.values():
                tot[k] = tot.get(k, 0) + int(r[k])
    s = sum(tot.values()) or 1
    print(f"DONE {ds}: {ok} bags. mix: " + "  ".join(f"{k} {100*v/s:.1f}%" for k, v in tot.items()))


if __name__ == "__main__":
    main()
