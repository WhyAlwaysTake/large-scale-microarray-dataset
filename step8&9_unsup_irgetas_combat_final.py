"""
ComBat Pipeline — Unsupervised (Train + Test)
==========================================================
Produces three H5 output files matching the input schema:
    train_normalized.h5   — ComBat-corrected train expression
    test_normalized.h5    — ComBat-corrected test expression (params from train)
    test_raw.h5           — Raw (uncorrected) test expression

Parameters are saved as a single H5 file:
    unsup_combat_params.h5
        /meta/settings          str   (4, 2)  [["chunk_size","2000"], ...]
        /chunk_0/
            slice_start         int scalar
            slice_stop          int scalar
            good_rows           bool  (chunk_g,)
            has_estimates       int scalar  (0 or 1)
            gamma_hat           float64  (n_batches, n_good)
            delta_hat           float64  (n_batches, n_good)
            gamma_bar           float64  (n_batches,)
            t2                  float64  (n_batches,)
            a_prior             float64  (n_batches,)
            b_prior             float64  (n_batches,)
            batches             str      (n_batches,)
            var.pooled          float64  (n_good, 1)
            stand.mean          float64  (n_good, n_train_samples)
            mod.mean            float64  (n_good, n_train_samples)
            gamma.star          float64  (n_batches, n_good)
            delta.star          float64  (n_batches, n_good)
        /chunk_1/
            ...

Input H5 schema  (samples × genes):
    expressions   float32  (n_samples, n_genes)
    gsm_ids       str      (n_samples,)
    gene_names    str      (n_genes,)

Output H5 schema (same layout):
    expressions    float32  (n_samples, n_genes)
    gsm_ids        str      (n_samples,)
    gene_names     str      (n_genes,)
    gse_ids        str      (n_samples,)
    disease_labels str      (n_samples,)
    icd_codes      str      (n_samples,)

CSV inputs:
    train.csv / test.csv  [GSM_ID, GSE_ID, icd_code, final_category]

Requirements:
    pip install h5py numpy pandas scipy neuroCombat tqdm
"""

import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from neuroCombat import neuroCombat, neuroCombatFromTraining

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TEST_MODE    = False
SUBSET_GENES = 5

INPUT_H5       = "/shared/home/rakhat.myrzakhan/new_combat/X_frma_deduped_jetset.h5"

TRAIN_CSV      = "/shared/home/rakhat.myrzakhan/new_combat/train.csv"
TEST_CSV       = "/shared/home/rakhat.myrzakhan/new_combat/test.csv"

OUT_TRAIN_NORM = "/shared/home/rakhat.myrzakhan/new_combat/unsup/train_normalized.h5"
OUT_TEST_NORM  = "/shared/home/rakhat.myrzakhan/new_combat/unsup/test_normalized.h5"
OUT_TEST_RAW   = "/shared/home/rakhat.myrzakhan/new_combat/unsup/test_raw.h5"
OUT_PARAMS_H5  = "/shared/home/rakhat.myrzakhan/new_combat/unsup/unsup_combat_params.h5"

KEY_EXPR  = "expressions"
KEY_GSM   = "gsm_ids"
KEY_GENES = "gene_names"

CHUNK_SIZE = 2000
MEAN_ONLY  = False
REF_BATCH  = None

COL_GSM      = "GSM_ID"
COL_BATCH    = "GSE_ID"
COL_CATEGORY = "final_category"
COL_ICD      = "icd_code"

# All estimate keys neuroCombat stores per chunk
ESTIMATE_KEYS = [
    "gamma_hat", "delta_hat", "gamma_bar", "t2",
    "a_prior",   "b_prior",   "batches",
    "var.pooled", "stand.mean", "mod.mean",
    "gamma.star", "delta.star",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    bar = "═" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def decode(arr):
    return [v.decode("utf-8", errors="replace") if isinstance(v, bytes)
            else str(v) for v in arr]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_labels():
    section("STEP 1 — Load labels")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    for df in (train, test):
        df[COL_GSM] = df[COL_GSM].astype(str).str.strip()

    train = train.set_index(COL_GSM)
    test  = test.set_index(COL_GSM)

    for name, df in [("train", train), ("test", test)]:
        before = len(df)
        df = df.loc[~df.index.duplicated(keep="first")]
        if len(df) < before:
            print(f"  [INFO] Removed {before - len(df):,} duplicate GSM_IDs from {name}")
        if name == "train": train = df
        else:               test  = df

    # No category count printed — unsupervised does not use disease labels
    print(f"  Train : {len(train):,} samples | {train[COL_BATCH].nunique():,} batches")
    print(f"  Test  : {len(test):,} samples  | {test[COL_BATCH].nunique():,} batches")
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — LOAD EXPRESSION FROM H5
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_rows(dset, row_indices: list, n_genes: int,
                desc: str = "") -> np.ndarray:
    H5_ROW_CHUNK = 2_000
    unique_rows  = sorted(set(row_indices))
    row_to_buf   = {r: i for i, r in enumerate(unique_rows)}
    buf          = np.empty((len(unique_rows), n_genes), dtype=np.float32)

    min_r, max_r = unique_rows[0], unique_rows[-1]
    for chunk_start in tqdm(range(min_r, max_r + 1, H5_ROW_CHUNK),
                            desc=f"  H5 read [{desc}]", leave=False):
        chunk_end = min(chunk_start + H5_ROW_CHUNK, max_r + 1)
        wanted    = [r for r in unique_rows if chunk_start <= r < chunk_end]
        if not wanted:
            continue
        block = dset[chunk_start:chunk_end, :n_genes].astype(np.float32)
        for r in wanted:
            buf[row_to_buf[r]] = block[r - chunk_start]

    out = np.empty((len(row_indices), n_genes), dtype=np.float32)
    for i, r in enumerate(row_indices):
        out[i] = buf[row_to_buf[r]]
    return out.T   # → (n_genes, n_samples)


def load_expression(train_lbl: pd.DataFrame, test_lbl: pd.DataFrame):
    section("STEP 2 — Load expression from H5")

    with h5py.File(INPUT_H5, "r") as f:
        for key in (KEY_EXPR, KEY_GSM, KEY_GENES):
            if key not in f:
                raise KeyError(f"Dataset '{key}' not found. "
                               f"Available: {list(f.keys())}")

        n_samples_h5, n_genes_h5 = f[KEY_EXPR].shape
        n_genes     = SUBSET_GENES if TEST_MODE else n_genes_h5
        h5_gsm_list = decode(f[KEY_GSM][:])
        gsm_to_row  = {gsm: i for i, gsm in enumerate(h5_gsm_list)}
        gene_names  = np.array(decode(f[KEY_GENES][:n_genes]))

        if TEST_MODE:
            print(f"  [TEST MODE] Using first {n_genes:,} / {n_genes_h5:,} genes")
        else:
            print(f"  H5 shape  : {n_samples_h5:,} samples × {n_genes_h5:,} genes")

        def resolve(labels_df):
            rows, ids = [], []
            for gsm in labels_df.index:
                row = gsm_to_row.get(gsm)
                if row is not None:
                    rows.append(row); ids.append(gsm)
            return rows, ids

        train_rows, train_ids = resolve(train_lbl)
        test_rows,  test_ids  = resolve(test_lbl)

        n_miss_tr = len(train_lbl) - len(train_ids)
        n_miss_te = len(test_lbl)  - len(test_ids)
        if n_miss_tr: print(f"  [WARN] {n_miss_tr:,} train GSM_IDs not in H5 — skipped")
        if n_miss_te: print(f"  [WARN] {n_miss_te:,} test  GSM_IDs not in H5 — skipped")

        dset = f[KEY_EXPR]
        print(f"\n  Fetching train ({len(train_ids):,} samples)...")
        train_mat = _fetch_rows(dset, train_rows, n_genes, desc="train")

        print(f"  Fetching test  ({len(test_ids):,} samples)...")
        test_mat  = _fetch_rows(dset, test_rows,  n_genes, desc="test")

    train_df = pd.DataFrame(train_mat, index=gene_names, columns=train_ids)
    test_df  = pd.DataFrame(test_mat,  index=gene_names, columns=test_ids)

    train_lbl_out = train_lbl.loc[train_df.columns]
    test_lbl_out  = test_lbl.loc[test_df.columns]

    assert train_df.shape[1] == len(train_lbl_out)
    assert test_df.shape[1]  == len(test_lbl_out)

    print(f"\n  Train : {train_df.shape[0]:,} genes × {train_df.shape[1]:,} samples")
    print(f"  Test  : {test_df.shape[0]:,} genes × {test_df.shape[1]:,} samples")
    return train_df, test_df, train_lbl_out, test_lbl_out, gene_names


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD COVARIATES — batch only, no biological covariate
# ══════════════════════════════════════════════════════════════════════════════

def build_covariates(labels_df: pd.DataFrame):
    """
    Returns a single-column [batch] covariate DataFrame.
    No disease/category covariate — this is unsupervised ComBat.
    """
    covars = pd.DataFrame(index=labels_df.index)
    covars["batch"] = labels_df[COL_BATCH].values
    return covars


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FIT UNSUPERVISED ComBat ON TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def fit_combat_chunked(expr_df: pd.DataFrame, covars_df: pd.DataFrame):
    """
    Unsupervised neuroCombat fit in CHUNK_SIZE-gene blocks.
    Only batch is modelled — no categorical or continuous biological covariates.
    Working layout: (genes × samples).

    Returns
    -------
    corrected_df : pd.DataFrame  (genes × samples)
    chunk_info   : list[dict]    per-chunk slice, good_rows mask, estimates
    """
    section(f"STEP 3 — Fit unsupervised ComBat on train  "
            f"({expr_df.shape[0]:,} genes × {expr_df.shape[1]:,} samples)")

    n_genes      = expr_df.shape[0]
    dat          = expr_df.values.astype(np.float32)
    covars_reset = covars_df.reset_index(drop=True)   # plain RangeIndex for neuroCombat

    corrected_full = dat.copy()
    chunk_info     = []
    n_ok = n_fail  = 0

    for i in tqdm(range(0, n_genes, CHUNK_SIZE), desc="  Train chunks"):
        end_i = min(i + CHUNK_SIZE, n_genes)
        chunk = dat[i:end_i, :]
        good  = np.var(chunk, axis=1) > 0      # skip zero-variance genes

        info = {"slice": slice(i, end_i), "good_rows": good, "estimates": None}

        if not np.any(good):
            chunk_info.append(info)
            continue

        sub = chunk[good].astype(np.float64)

        try:
            res = neuroCombat(
                dat              = sub,
                covars           = covars_reset,
                batch_col        = "batch",
                categorical_cols = [],          # ← unsupervised: no biological covariate
                continuous_cols  = [],
                mean_only        = MEAN_ONLY,
                ref_batch        = REF_BATCH,
            )
            corrected_full[i:end_i][good] = res["data"].astype(np.float32)
            info["estimates"] = res["estimates"]
            n_ok += 1
        except Exception as e:
            print(f"\n    [WARN] Chunk {i}:{end_i} — {e}  (raw values kept)")
            n_fail += 1

        chunk_info.append(info)

    print(f"\n  Chunks succeeded : {n_ok:,} / {len(chunk_info):,}  ({n_fail:,} failed)")

    if n_ok == 0:
        raise RuntimeError(
            "ComBat failed for every gene chunk. "
            "Verify that batch labels align with expression columns and "
            "that genes have non-zero variance.")

    corrected_df = pd.DataFrame(corrected_full,
                                index=expr_df.index,
                                columns=expr_df.columns)
    return corrected_df, chunk_info


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — APPLY FROZEN PARAMS TO TEST
# ══════════════════════════════════════════════════════════════════════════════

def apply_combat_chunked(expr_df: pd.DataFrame,
                         covars_df: pd.DataFrame,
                         chunk_info: list):
    """
    Apply per-chunk unsupervised train estimates to test via
    neuroCombatFromTraining. Samples from unseen batches are dropped.

    Returns
    -------
    corrected_df : pd.DataFrame  (genes × samples)
    """
    section("STEP 4 — Apply frozen params to test")

    def _str_set(it):
        return {v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in it}

    covars_df = covars_df.copy()
    covars_df["batch"] = covars_df["batch"].astype(str)

    first_est = next((c["estimates"] for c in chunk_info
                      if c["estimates"] is not None), None)
    if first_est is None:
        raise RuntimeError("No valid chunk estimates found.")

    train_batches = _str_set(first_est["batches"])
    test_batches  = _str_set(covars_df["batch"].unique())
    unseen        = test_batches - train_batches

    if unseen:
        n_drop    = int(covars_df["batch"].isin(unseen).sum())
        print(f"  Dropping {n_drop:,} samples from {len(unseen):,} unseen batches")
        keep      = ~covars_df["batch"].isin(unseen)
        covars_df = covars_df[keep]
        expr_df   = expr_df[covars_df.index]

    print(f"  Known train batches : {len(train_batches):,}")
    print(f"  Test batches kept   : {len(test_batches) - len(unseen):,}")
    print(f"  Test samples kept   : {expr_df.shape[1]:,}")

    covars_reset   = covars_df.reset_index(drop=True)
    dat            = expr_df.values.astype(np.float32)
    corrected_full = dat.copy()

    for info in tqdm(chunk_info, desc="  Test apply"):
        sl, good, estimates = info["slice"], info["good_rows"], info["estimates"]

        if estimates is None or not np.any(good):
            continue

        sub = dat[sl][good].astype(np.float64)

        estimates_patched = {
            **estimates,
            "batches": [b.decode("utf-8") if isinstance(b, bytes) else str(b)
                        for b in estimates["batches"]],
        }

        try:
            res = neuroCombatFromTraining(
                dat       = sub,
                batch     = covars_reset["batch"].values,
                estimates = estimates_patched,
            )
            corrected_full[sl][good] = res["data"].astype(np.float32)
        except Exception as e:
            print(f"\n    [WARN] Apply chunk {sl} — {e}  (raw values kept)")

    corrected_df = pd.DataFrame(corrected_full,
                                index=expr_df.index,
                                columns=expr_df.columns)
    return corrected_df


# ══════════════════════════════════════════════════════════════════════════════
#  SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(original_df, corrected_df, tag=""):
    print(f"\n── Sanity check [{tag}] {'─'*40}")
    g, s = corrected_df.shape
    print(f"  Shape       : {g:,} genes × {s:,} samples")
    print(f"  Global mean : {original_df.values.mean():.4f}  →  "
          f"{corrected_df.values.mean():.4f}")
    n_nan = int(np.isnan(corrected_df.values).sum())
    n_inf = int(np.isinf(corrected_df.values).sum())
    print(f"  NaNs / Infs : {n_nan:,} / {n_inf:,}  "
          f"[{'OK' if not (n_nan or n_inf) else '!! WARNING'}]")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE H5 OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_h5(expr_df, labels_df, gene_names, path, tag):
    print(f"\n  Writing [{tag}]  →  {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    gsm_ids   = expr_df.columns.tolist()
    lbl       = labels_df.loc[gsm_ids]
    n_samples = len(gsm_ids)
    n_genes   = len(gene_names)
    data_out  = expr_df.values.T.astype(np.float32)   # (samples × genes)

    assert data_out.shape == (n_samples, n_genes), \
        f"Shape mismatch: expected ({n_samples}, {n_genes}), got {data_out.shape}"

    dt = h5py.special_dtype(vlen=str)
    with h5py.File(path, "w") as f:
        f.create_dataset(KEY_EXPR, data=data_out,
                         compression="gzip", compression_opts=4,
                         chunks=(min(500, n_samples), min(1_000, n_genes)))
        f.create_dataset(KEY_GSM,          data=np.array(gsm_ids, dtype=dt))
        f.create_dataset(KEY_GENES,        data=np.array(gene_names, dtype=dt))
        f.create_dataset("gse_ids",        data=np.array(lbl[COL_BATCH].tolist(),    dtype=dt))
        f.create_dataset("disease_labels", data=np.array(lbl[COL_CATEGORY].tolist(), dtype=dt))
        f.create_dataset("icd_codes",      data=np.array(lbl[COL_ICD].tolist(),      dtype=dt))

    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"    {n_samples:,} samples × {n_genes:,} genes  |  {size_mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE PARAMS — single H5, one group per chunk
#  No label_map stored (no disease covariate in unsupervised mode)
# ══════════════════════════════════════════════════════════════════════════════

def save_params(chunk_info: list, path: str):
    """
    Save all unsupervised ComBat parameters into a single H5 file.

    Layout:
        /meta/settings      str  (4, 2)  [["chunk_size","2000"], ...]
        /chunk_0/
            slice_start     int scalar
            slice_stop      int scalar
            good_rows       uint8  (chunk_g,)
            has_estimates   int scalar
            gamma_hat       float64  (n_batches, n_good)   compressed
            ...             (all ESTIMATE_KEYS)
        /chunk_1/ ...
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dt = h5py.special_dtype(vlen=str)

    with h5py.File(path, "w") as f:

        # ── /meta — settings only (no label_map for unsupervised) ─────────────
        meta_grp     = f.create_group("meta")
        settings_arr = np.array([
            ["chunk_size", str(CHUNK_SIZE)],
            ["mean_only",  str(MEAN_ONLY)],
            ["ref_batch",  str(REF_BATCH)],
            ["mode",       "unsupervised"],
        ], dtype="S")
        meta_grp.create_dataset("settings", data=settings_arr)

        # ── /chunk_N ──────────────────────────────────────────────────────────
        for idx, c in enumerate(tqdm(chunk_info, desc="  Saving chunks")):
            grp = f.create_group(f"chunk_{idx}")
            grp.create_dataset("slice_start",   data=c["slice"].start)
            grp.create_dataset("slice_stop",    data=c["slice"].stop)
            grp.create_dataset("good_rows",     data=c["good_rows"].astype(np.uint8))
            has_est = c["estimates"] is not None
            grp.create_dataset("has_estimates", data=int(has_est))

            if not has_est:
                continue

            est = c["estimates"]
            for k in ESTIMATE_KEYS:
                if k not in est:
                    continue
                v = est[k]
                if k == "batches":
                    batch_arr = np.array(
                        [b.decode("utf-8") if isinstance(b, bytes) else str(b)
                         for b in v],
                        dtype=dt,
                    )
                    grp.create_dataset(k, data=batch_arr)
                else:
                    grp.create_dataset(
                        k, data=np.array(v, dtype=np.float64),
                        compression="gzip", compression_opts=4,
                    )

    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"\n  Saved unsup_combat_params.h5  →  {path}  ({size_mb:.1f} MB)")
    print(f"  Chunks : {len(chunk_info)}  |  "
          f"Estimate keys per chunk : {len(ESTIMATE_KEYS)}")


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PARAMS — reconstruct chunk_info from H5
# ══════════════════════════════════════════════════════════════════════════════

def load_params(path: str):
    """
    Reconstruct chunk_info from unsup_combat_params.h5.

    Returns
    -------
    chunk_info : list[dict]   {slice, good_rows, estimates}
    """
    section("Load saved unsupervised ComBat parameters")

    with h5py.File(path, "r") as f:
        settings_raw = f["meta/settings"][:]
        settings = {
            (r[0].decode("utf-8") if isinstance(r[0], bytes) else r[0]):
            (r[1].decode("utf-8") if isinstance(r[1], bytes) else r[1])
            for r in settings_raw
        }
        print(f"  Mode       : {settings.get('mode', '?')}")
        print(f"  Chunk size : {settings.get('chunk_size', '?')}")

        chunk_keys = sorted(
            [k for k in f.keys() if k.startswith("chunk_")],
            key=lambda x: int(x.split("_")[1]),
        )
        print(f"  N chunks   : {len(chunk_keys)}")

        chunk_info = []
        for ck in tqdm(chunk_keys, desc="  Loading chunks"):
            grp       = f[ck]
            sl        = slice(int(grp["slice_start"][()]),
                              int(grp["slice_stop"][()]))
            good_rows = grp["good_rows"][:].astype(bool)
            has_est   = bool(int(grp["has_estimates"][()]))

            if has_est:
                estimates = {}
                for k in ESTIMATE_KEYS:
                    if k not in grp:
                        continue
                    raw = grp[k][:]
                    if k == "batches":
                        estimates[k] = [
                            v.decode("utf-8") if isinstance(v, bytes) else str(v)
                            for v in raw
                        ]
                    else:
                        estimates[k] = raw.astype(np.float64)
            else:
                estimates = None

            chunk_info.append({
                "slice":     sl,
                "good_rows": good_rows,
                "estimates": estimates,
            })

    n_ok = sum(1 for c in chunk_info if c["estimates"] is not None)
    print(f"  Chunks with estimates : {n_ok} / {len(chunk_info)}")
    return chunk_info


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if TEST_MODE:
        section(f"!! TEST MODE  —  first {SUBSET_GENES:,} genes only !!")

    # 1. Labels
    train_lbl, test_lbl = load_labels()

    # 2. Expression
    (train_expr, test_expr,
     train_lbl, test_lbl,
     gene_names) = load_expression(train_lbl, test_lbl)

    test_expr_raw = test_expr.copy()

    # 3. Covariates — batch only (no disease label for unsupervised)
    train_covars = build_covariates(train_lbl)
    test_covars  = build_covariates(test_lbl)

    # 4. Fit unsupervised ComBat on train
    train_corrected, chunk_info = fit_combat_chunked(train_expr, train_covars)

    # 5. Apply to test
    test_corrected = apply_combat_chunked(test_expr, test_covars, chunk_info)

    test_lbl_final = test_lbl.loc[test_corrected.columns]

    # 6. Sanity checks
    sanity_check(train_expr,    train_corrected,                              "TRAIN normalised")
    sanity_check(test_expr_raw, test_corrected,                               "TEST  normalised")
    sanity_check(test_expr_raw, test_expr_raw.loc[:, test_corrected.columns], "TEST  raw       ")

    # 7. Save outputs
    section("STEP 5 — Save outputs")

    save_h5(train_corrected, train_lbl,       gene_names, OUT_TRAIN_NORM, "train_normalized")
    save_h5(test_corrected,  test_lbl_final,  gene_names, OUT_TEST_NORM,  "test_normalized")
    save_h5(test_expr_raw.loc[:, test_corrected.columns],
            test_lbl_final,  gene_names, OUT_TEST_RAW,   "test_raw")

    save_params(chunk_info, OUT_PARAMS_H5)

    print(f"\n{'═'*62}")
    print(f"  DONE{'  [TEST MODE]' if TEST_MODE else ''}")
    print(f"{'═'*62}")
