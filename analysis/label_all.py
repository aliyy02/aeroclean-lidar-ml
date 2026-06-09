"""Batch-label every Oxy + Bech bag with the validated frame conventions.

alpha=24, ysign=+1 (axis map z,x,y) -- validated: fitted-facade-normal matches GT
normal to <1 deg median across tilted poses for both datasets.
"""
import sys
import os
sys.path.insert(0, ".")
from calibration.label_scan import label_scan   # noqa: E402

DS = [
    ("Oxy_L2_All/Oxy_L2_test_{n}", "Oxy_L2_Text_All/Oxy_L2_Text_All/Oxy_L2_test_{n}.txt",
     "data/labeled_oxy/Oxy_{n}", range(1, 60)),
    ("Bech_all/Bech_109_test_{n}", "Bech_Text_All/Bech_Text_All/Bech_109_test_{n}.txt",
     "data/labeled_bech/Bech_{n}", range(1, 41)),
]

ok = bad = 0
for bagt, gtt, outt, ns in DS:
    for n in ns:
        bag, gt, out = bagt.format(n=n), gtt.format(n=n), outt.format(n=n)
        if not os.path.isdir(bag) or not os.path.isfile(gt):
            continue
        try:
            label_scan(bag, gt, out, alpha=24.0, qa=False)
            ok += 1
        except Exception as e:
            print(f"FAIL {bag}: {e}")
            bad += 1
print(f"DONE: {ok} labeled, {bad} failed")
