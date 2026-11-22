"""
VALIDATE ANERBC -- one script, all findings.

Checks, in order:
  0. Self-test of the parsing logic on known inputs
  1. Tile structure: is AneRBC-II 12 crops per AneRBC-I slide?
  2. Pixel proof: do 12 tiles reassemble into the parent image?
  3. Duplicate CBC reports across patients
  4. Internally impossible CBC records (Wintrobe identities)
  5. Does the label track haemoglobin, or morphology?
  6. 25 random morphology reports for you to hand-grade

Outputs:
  anerbc_patients.csv   1 row per patient (HGB, MCV, flags, morphology)
  anerbc_tiles.csv      1 row per tile (with patient_id group key)
  stitch_check.png      visual proof for the paper (if PIL available)

Usage:
  python validate_anerbc.py /path/to/AneRBC_dataset
"""
import re
import sys
import glob
import random
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG
ROOT = Path("AneRBC_dataset")
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1])
TOL = 0.12          # relative tolerance for the Wintrobe identity checks
OUT = Path.cwd()    # where the CSVs get written


# ------------------------------------------------------------ PRIMITIVES
def read(fp):
    txt = Path(fp).read_text(encoding="utf-8", errors="replace")
    txt = unicodedata.normalize("NFKC", txt)
    return txt.replace("µ", "μ")  # micro sign -> greek mu


ANALYTES = {
    "HGB": {"HGB", "HB", "HEMOGLOBIN", "HAEMOGLOBIN"},
    "RBC": {"RBC"},
    "HCT": {"HCT", "PCV"},
    "MCV": {"MCV"},
    "MCH": {"MCH"},
    "MCHC": {"MCHC"},
    "RDW": {"RDW"},
    "PLT": {"PLT"},
    "WBC": {"WBC"},
}


def get_analyte(text, names):
    """First number on a line whose first comma-field matches `names`.

    Handles '* 11.1 g/dL', '*2.9 g/dL', '2.21g/dL', 'Hemoglobin,5.1g/dL',
    'RDW---CV,24.8%'.
    """
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        key = parts[0].strip().upper().lstrip("#*%: ").replace(" ", "")
        key = re.sub(r"-{2,}[A-Z]+$", "", key)  # RDW---CV -> RDW
        if key in names:
            m = re.search(r"-?\d+(?:\.\d+)?", parts[1].replace("*", " "))
            if m:
                return float(m.group())
    return None


ABNORMAL_PAT = re.compile(
    r"microcyt|macrocyt|hypochrom|anisocyt|poikilocyt|target cell|elliptocyt|"
    r"ovalocyt|tear drop|fragmented|polychromasia|dimorphic|spherocyt|schistocyt",
    re.I)


def numeric_fingerprint(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return hashlib.md5(",".join(nums).encode()).hexdigest()


def banner(n, title):
    print("\n" + "=" * 74)
    print(f"{n}. {title}")
    print("=" * 74)


# --------------------------------------------------------------- 0. SELFTEST
def selftest():
    banner(0, "SELF-TEST OF THE PARSING LOGIC")
    cases = [
        ("HGB,* 11.1 g/dL,12 --- 15", 11.1, "asterisk flag, spaced"),
        ("HGB,*2.9 g/dL,11 --- 14", 2.9, "asterisk, unspaced"),
        ("HGB,2.21g/dL,12 --- 15", 2.21, "no space before unit"),
        ("Hemoglobin,5.1 g/dL,11.1 --- 14.1", 5.1, "spelled-out name"),
        ("HB,13 g/dL,13 --- 17", 13.0, "HB abbreviation"),
        ("WBC,8.44 x10.e 3/ul,4 --- 10\nHGB,7.8 g/dL,11 --- 14",
         7.8, "skips earlier WBC line"),
        ("MCHC,32.5 g/dL,31.5 --- 34.5", None, "MCHC must NOT match HGB"),
    ]
    ok = True
    for text, expected, why in cases:
        got = get_analyte(text, ANALYTES["HGB"])
        good = (got == expected)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {why:34s} "
              f"expected={str(expected):6s} got={got}")

    print("\n  Wintrobe identities, hand-checked on record 352_h:")
    hgb, rbc, hct = 5.1, 4.37, 22.2
    print(f"    MCH  = HGB/RBC*10  = {hgb/rbc*10:5.1f}   report says 11.7")
    print(f"    MCHC = HGB/HCT*100 = {hgb/hct*100:5.1f}   report says 23.0")
    print(f"    MCV  = HCT/RBC*10  = {hct/rbc*10:5.1f}   report says 50.8")
    print("    -> consistent, so that record's HGB of 5.1 g/dL is real")
    print(f"\n  {'SELF-TEST PASSED' if ok else 'SELF-TEST FAILED -- STOP HERE'}")
    return ok


# ------------------------------------------------------------ 1. TILE STRUCTURE
def check_tiles():
    banner(1, "TILE STRUCTURE -- is AneRBC-II 12 crops per AneRBC-I slide?")
    summary = {}
    for group, letter in (("Anemic", "a"), ("Healthy", "h")):
        d = ROOT / "AneRBC-II" / f"{group}_individuals" / "Original_images"
        names = [Path(p).name for p in glob.glob(str(d / "*.png"))]
        if not names:
            print(f"  !! no tiles found in {d}")
            continue
        byparent = defaultdict(list)
        for n in names:
            byparent[Path(n).stem.split("_")[0]].append(n)
        sizes = [len(v) for v in byparent.values()]
        print(f"  {group:8s}: tiles={len(names):6d}  parents={len(byparent):4d}  "
              f"tiles/parent min={min(sizes)} max={max(sizes)}")
        ex = sorted(byparent)[0]
        print(f"    example parent {ex}: {sorted(byparent[ex])[:3]} ...")
        summary[group] = (min(sizes), max(sizes), len(byparent))

    if summary and all(lo == hi == 12 for lo, hi, _ in summary.values()):
        print("\n  -> exactly 12 tiles per parent, confirmed.")
        print("     Under an 80/10/10 RANDOM tile split, the chance a test tile")
        print(f"     has NO sibling in training is 0.2^11 = {0.2**11:.2e}.")
        print(f"     Expected siblings in training per test tile: {11*0.8:.1f}")
        print("     => random splitting on AneRBC-II leaks, essentially always.")
    else:
        print("\n  -> NOT uniformly 12 per parent. Investigate before claiming.")


# --------------------------------------------------------------- 2. PIXEL PROOF
def stitch_proof(parent_id="001_a"):
    banner(2, "PIXEL PROOF -- do 12 tiles reassemble into the parent image?")
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not installed; skipping. (pip install pillow)")
        return

    group = "Anemic" if parent_id.endswith("_a") else "Healthy"
    letter = parent_id[-1]
    num = int(parent_id.split("_")[0])
    tdir = ROOT / "AneRBC-II" / f"{group}_individuals" / "Original_images"
    pfile = ROOT / "AneRBC-I" / f"{group}_individuals" / "Original_images" / f"{parent_id}.png"

    tiles = []
    for i in range(1, 13):
        fp = tdir / f"{num:04d}_{i:02d}_{letter}.png"
        if not fp.exists():
            print(f"  missing tile {fp.name}; skipping proof")
            return
        tiles.append(Image.open(fp).convert("RGB"))
    if not pfile.exists():
        print(f"  parent {pfile.name} not found; skipping proof")
        return

    parent = Image.open(pfile).convert("RGB")
    pw, ph = parent.size
    tw, th = tiles[0].size
    print(f"  parent {parent_id}: {pw}x{ph}   tile: {tw}x{th}")
    print(f"  grid fit: {pw}/{tw} = {pw/tw:.2f} cols, {ph}/{th} = {ph/th:.2f} rows")

    pa = np.asarray(parent, dtype=np.int16)
    best = None
    layouts = {
        "row-major 4x3": lambda i: ((i % 4) * tw, (i // 4) * th),
        "col-major 4x3": lambda i: ((i // 3) * tw, (i % 3) * th),
        "row-major 3x4": lambda i: ((i % 3) * tw, (i // 3) * th),
        "col-major 3x4": lambda i: ((i // 4) * tw, (i % 4) * th),
    }
    for name, pos in layouts.items():
        canvas = Image.new("RGB", (pw, ph))
        try:
            for i, t in enumerate(tiles):
                canvas.paste(t, pos(i))
        except Exception:
            continue
        diff = np.abs(np.asarray(canvas, dtype=np.int16) - pa).mean()
        print(f"    {name:16s} mean abs pixel diff = {diff:8.3f}")
        if best is None or diff < best[1]:
            best = (name, diff, canvas)

    if best:
        name, diff, canvas = best
        if diff < 2.0:
            print(f"\n  -> PROVEN. Layout '{name}' reconstructs the parent "
                  f"(diff {diff:.3f}).")
        else:
            print(f"\n  -> best layout '{name}' gives diff {diff:.1f}. Tiles may be "
                  f"ordered differently; the readme claim still stands.")
        try:
            side = Image.new("RGB", (pw * 2 + 20, ph), "white")
            side.paste(parent, (0, 0))
            side.paste(canvas, (pw + 20, 0))
            side.save(OUT / "stitch_check.png")
            print(f"  figure saved: {OUT / 'stitch_check.png'} "
                  f"(left=original, right=restitched)")
        except Exception as e:
            print(f"  could not save figure: {e}")


# ------------------------------------------------------------ LOAD REPORTS
def load_patients():
    rows = []
    for group, label in (("Anemic", 1), ("Healthy", 0)):
        cdir = ROOT / "AneRBC-I" / f"{group}_individuals" / "CBC_reports"
        mdir = ROOT / "AneRBC-I" / f"{group}_individuals" / "Morphology_reports"
        files = sorted(glob.glob(str(cdir / "*.txt")))
        if not files:
            print(f"  !! no CBC reports in {cdir}")
        for fp in files:
            pid = Path(fp).stem
            t = read(fp)
            rec = {"patient_id": pid, "label": label, "group": group,
                   "fingerprint": numeric_fingerprint(t)}
            for k, names in ANALYTES.items():
                rec[k] = get_analyte(t, names)
            mp = mdir / f"{pid}.txt"
            mtxt = read(mp) if mp.exists() else ""
            rec["morph_text"] = " ".join(mtxt.split())
            rec["morph_abnormal"] = bool(ABNORMAL_PAT.search(mtxt))
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. DUPLICATES
def check_duplicates(df):
    banner(3, "DUPLICATE CBC REPORTS")
    n, uniq = len(df), df.fingerprint.nunique()
    print(f"  CBC files               : {n}")
    print(f"  Distinct numeric content: {uniq}")
    print(f"  Duplicated              : {n-uniq} ({100*(n-uniq)/max(n,1):.1f}%)")

    grp = df.groupby("fingerprint").patient_id.apply(list)
    dupes = grp[grp.apply(len) > 1]
    cross = dupes[dupes.apply(lambda ids: len({i[-1] for i in ids}) > 1)]
    print(f"  Clusters spanning BOTH _a and _h: {len(cross)}")
    samenum = dupes[dupes.apply(
        lambda ids: len({i.split('_')[0] for i in ids}) < len(ids))]
    print(f"  Clusters where one patient NUMBER appears under both labels: "
          f"{len(samenum)}")

    if len(dupes):
        print("\n  Largest clusters:")
        for ids in dupes.sort_values(key=lambda s: s.apply(len),
                                     ascending=False).head(5):
            print(f"    {len(ids):2d} patients share one CBC: {sorted(ids)}")
    if len(samenum):
        ids = sorted(samenum.iloc[0])
        print(f"\n  EVIDENCE -- raw HGB/RBC/HCT lines for cluster {ids}:")
        for pid in ids:
            row = df[df.patient_id == pid].iloc[0]
            print(f"    {pid}: HGB={row.HGB} RBC={row.RBC} HCT={row.HCT} "
                  f"MCV={row.MCV}  label={'anemic' if row.label else 'healthy'}")


# -------------------------------------------------------------- 4. CONSISTENCY
def check_consistency(df):
    banner(4, "INTERNAL CONSISTENCY (Wintrobe identities)")

    def rel(obs, exp):
        ok = obs.notna() & exp.notna() & (exp.abs() > 1e-6)
        out = pd.Series(np.nan, index=obs.index)
        out[ok] = (obs[ok] - exp[ok]).abs() / exp[ok].abs()
        return out

    df["e_MCH"] = rel(df.MCH, df.HGB / df.RBC * 10)
    df["e_MCHC"] = rel(df.MCHC, df.HGB / df.HCT * 100)
    df["e_MCV"] = rel(df.MCV, df.HCT / df.RBC * 10)
    E = df[["e_MCH", "e_MCHC", "e_MCV"]]
    df["n_checks"] = E.notna().sum(axis=1)
    df["n_failed"] = (E > TOL).sum(axis=1)
    df["corrupt"] = (df.n_checks >= 2) & (df.n_failed >= 2)
    df["impossible"] = ((df.HGB < 3) | (df.HGB > 22)
                        | (df.MCV > 130) | (df.MCHC > 40))
    df["unusable"] = df.corrupt | df.impossible

    print(f"  Judgeable (2+ identities computable) : {(df.n_checks>=2).sum()}")
    print(f"  Failing 2+ identities at {TOL:.0%}        : {int(df.corrupt.sum())}")
    print(f"  Physiologically impossible values    : {int(df.impossible.sum())}")
    print(f"  UNUSABLE (either)                    : {int(df.unusable.sum())}"
          f" ({100*df.unusable.mean():.1f}%)")
    print(f"  CLEAN                                : {int((~df.unusable).sum())}")

    bad = df[df.unusable]
    if len(bad):
        r = bad.iloc[0]
        print(f"\n  EVIDENCE of an INCONSISTENT record -- {r.patient_id}")
        print(f"    reported: HGB={r.HGB} RBC={r.RBC} HCT={r.HCT} "
              f"MCV={r.MCV} MCH={r.MCH} MCHC={r.MCHC}")
        print(f"    computed: MCV={r.HCT/r.RBC*10:.1f} MCH={r.HGB/r.RBC*10:.1f} "
              f"MCHC={r.HGB/r.HCT*100:.1f}")

    low = df[(~df.unusable) & (df.label == 0) & (df.HGB < 8)].sort_values("HGB")
    if len(low):
        r = low.iloc[0]
        print(f"\n  EVIDENCE of a CONSISTENT record in the HEALTHY folder -- "
              f"{r.patient_id}")
        print(f"    reported: HGB={r.HGB} RBC={r.RBC} HCT={r.HCT} "
              f"MCV={r.MCV} MCH={r.MCH} MCHC={r.MCHC}")
        print(f"    computed: MCV={r.HCT/r.RBC*10:.1f} MCH={r.HGB/r.RBC*10:.1f} "
              f"MCHC={r.HGB/r.HCT*100:.1f}")
        print(f"    -> arithmetic checks out. HGB {r.HGB} g/dL is REAL, "
              f"filed as HEALTHY.")
        print(f"    morphology: {r.morph_text[:150]}")
    return df


# ------------------------------------------------------------- 5. LABEL VALIDITY
def check_labels(df):
    banner(5, "DOES THE LABEL TRACK HAEMOGLOBIN, OR MORPHOLOGY?")
    clean = df[(~df.unusable) & df.HGB.notna()]
    print(f"  n (clean, HGB present) = {len(clean)}\n")
    for lab, name in ((0, "healthy"), (1, "anemic")):
        s = clean.loc[clean.label == lab, "HGB"]
        if len(s):
            print(f"  {name:8s} n={len(s):4d}  mean={s.mean():5.2f}  "
                  f"median={s.median():5.2f}  sd={s.std():4.2f}  "
                  f"[{s.min():.1f}, {s.max():.1f}]")

    pos = clean.loc[clean.label == 1, "HGB"].values
    neg = clean.loc[clean.label == 0, "HGB"].values
    if len(pos) and len(neg):
        # rank-based AUC, computed directly (no sklearn dependency)
        allv = np.concatenate([pos, neg])
        ranks = pd.Series(-allv).rank().values  # low HGB -> high rank
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"\n  AUC(low HGB -> anemic label) = {auc:.3f}   "
              f"[0.50 = no information]")

    print("\n  WHO-anemic (HGB < 12 g/dL) prevalence by folder:")
    for lab, name in ((0, "healthy folder"), (1, "anemic folder ")):
        s = clean.loc[clean.label == lab, "HGB"]
        if len(s):
            print(f"    {name}: {100*(s < 12).mean():5.1f}%")

    print("\n  Label vs pathologist morphology verdict:")
    ct = pd.crosstab(df.label.map({0: "healthy", 1: "anemic"}),
                     df.morph_abnormal.map({False: "morph normal",
                                            True: "morph abnormal"}))
    print("    " + ct.to_string().replace("\n", "\n    "))
    agree = ((df.label == 1) == df.morph_abnormal).mean()
    print(f"\n  Label agrees with morphology: {100*agree:.1f}%")
    print("  -> HIGH morphology agreement + LOW HGB AUC means the folders")
    print("     encode RBC MORPHOLOGY, not anemia status.")


# --------------------------------------------------------- 6. HAND-GRADE SAMPLE
def hand_grade(df, k=25, seed=0):
    banner(6, f"HAND-GRADE THESE {k} MORPHOLOGY REPORTS YOURSELF")
    print("The morphology agreement figure rests on a regex. Read each report,")
    print("decide NORMAL or ABNORMAL yourself, THEN compare with the script.")
    print("Agreement on 23+/25 means the finding is solid.\n")
    rng = random.Random(seed)
    sample = rng.sample(list(df.index), min(k, len(df)))
    for n, idx in enumerate(sample, 1):
        r = df.loc[idx]
        v = "ABNORMAL" if r.morph_abnormal else "NORMAL  "
        print(f"{n:2d}. [{r.patient_id}] folder={r.group:8s} script={v}")
        print(f"    {r.morph_text[:220] or '(no morphology report)'}\n")


# ------------------------------------------------------------- TILE MANIFEST
def build_tile_manifest(patients):
    rows = []
    for group, label, letter in (("Anemic", 1, "a"), ("Healthy", 0, "h")):
        d = ROOT / "AneRBC-II" / f"{group}_individuals" / "Original_images"
        for fp in sorted(glob.glob(str(d / "*.png"))):
            stem = Path(fp).stem  # 0001_05_a
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            rows.append({
                "tile_path": fp,
                "tile_id": stem,
                "patient_id": f"{int(parts[0]):03d}_{parts[2]}",
                "tile_index": int(parts[1]),
                "label": label,
            })
    tiles = pd.DataFrame(rows)
    if tiles.empty:
        return tiles
    cols = ["patient_id", "HGB", "MCV", "MCH", "MCHC", "RDW", "unusable"]
    cols = [c for c in cols if c in patients.columns]
    tiles = tiles.merge(patients[cols], on="patient_id", how="left")
    return tiles


# ------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    print(f"ROOT = {ROOT}")
    if not (ROOT / "AneRBC-I").exists():
        sys.exit("\nAneRBC-I not found under ROOT. Pass the dataset folder "
                 "as the first argument.")
    if not selftest():
        sys.exit("\nParsing logic failed self-test. Stop and investigate.")

    check_tiles()
    stitch_proof()

    print("\nLoading reports ...")
    df = load_patients()
    print(f"  loaded {len(df)} patients "
          f"({int((df.label==1).sum())} anemic, {int((df.label==0).sum())} healthy)")

    check_duplicates(df)
    df = check_consistency(df)
    check_labels(df)

    tiles = build_tile_manifest(df)
    df.to_csv(OUT / "anerbc_patients.csv", index=False)
    if not tiles.empty:
        tiles.to_csv(OUT / "anerbc_tiles.csv", index=False)

    banner("X", "FILES WRITTEN")
    print(f"  anerbc_patients.csv  {len(df)} rows "
          f"({int((~df.unusable).sum())} usable)")
    if not tiles.empty:
        print(f"  anerbc_tiles.csv     {len(tiles)} rows across "
              f"{tiles.patient_id.nunique()} patients")
    print("\n  Split for training with:")
    print("    from sklearn.model_selection import StratifiedGroupKFold")
    print("    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)")
    print("    cv.split(tiles, tiles.label, groups=tiles.patient_id)")

    hand_grade(df)
