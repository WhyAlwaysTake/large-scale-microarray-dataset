"""
Probe → Gene Mapping (JetSet)
==========================================================
Input H5  (samples x probes):
    expressions   float32  (169234, 54675)
    gsm_ids       object   (169234,)
    probe_ids     object   (54675,)

Output H5 (samples x genes) — same structure, genes instead of probes:
    expressions   float32  (169234, n_genes)
    gsm_ids       object   (169234,)          ← copied as-is
    gene_names    object   (n_genes,)

Requirements:
    pip install h5py numpy pandas mygene matplotlib seaborn tqdm
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import mygene
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INPUT_H5    = "/X_frma_deduped.h5"
OUTPUT_H5   = "/X_frma_deduped_jetset.h5"
JETSET_CSV  = "/jetset.scores.hgu133plus2_3.4.0.csv"

CHUNK_SIZE  = 5_000   # samples processed per write iteration
SUBSET_SIZE = 2_000   # samples used to compute per-probe mean intensity (fallback)

# H5 dataset keys
KEY_EXPR    = "expressions"
KEY_GSM     = "gsm_ids"
KEY_PROBES  = "probe_ids"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD PROBES + MEAN INTENSITY FROM H5
# ══════════════════════════════════════════════════════════════════════════════

print("─" * 60)
print("  STEP 1 — Load probes & compute mean intensity")
print("─" * 60)

with h5py.File(INPUT_H5, "r") as f:
    # Validate expected keys
    for key in (KEY_EXPR, KEY_GSM, KEY_PROBES):
        if key not in f:
            raise KeyError(f"Expected dataset '{key}' not found in H5. "
                           f"Available keys: {list(f.keys())}")

    # Load probe IDs
    raw_probes = f[KEY_PROBES][:]
    all_probes = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in raw_probes
    ]
    n_probes = len(all_probes)

    # H5 layout: (samples, probes)
    n_samples_total, n_probes_h5 = f[KEY_EXPR].shape
    assert n_probes_h5 == n_probes, (
        f"Mismatch: probe_ids has {n_probes} entries but "
        f"expressions has {n_probes_h5} columns")

    print(f"  Probes   : {n_probes:,}")
    print(f"  Samples  : {n_samples_total:,}")

    # Compute per-probe mean over a subset of samples (used as tie-break)
    n_load      = min(n_samples_total, SUBSET_SIZE)
    subset_data = f[KEY_EXPR][:n_load, :]          # (n_load, n_probes)
    probe_means = np.mean(subset_data, axis=0)     # mean across samples → (n_probes,)
    print(f"  Mean intensity computed over first {n_load:,} samples")

probe_intensity_map = dict(zip(all_probes, probe_means))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MAP PROBES → GENE SYMBOLS (MyGene.info)
# ══════════════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print("  STEP 2 — Query MyGene.info for probe → gene mapping")
print("─" * 60)

mg      = mygene.MyGeneInfo()
results = mg.querymany(
    all_probes,
    scopes  = "reporter,accession,alias",
    fields  = "symbol",
    species = "human",
    verbose = False,
)

probe_index_map = {name: i for i, name in enumerate(all_probes)}

map_rows = []
for res in results:
    if "symbol" not in res or "query" not in res:
        continue
    p = res["query"]
    if p in probe_index_map:
        map_rows.append({
            "index":     probe_index_map[p],
            "probe":     p,
            "gene":      res["symbol"],
            "intensity": probe_intensity_map[p],
        })

df_map = pd.DataFrame(map_rows)

if df_map.empty:
    print("CRITICAL: No probes could be mapped to gene symbols. "
          "Check internet connection and probe ID format.")
    sys.exit(1)

print(f"  {len(df_map):,} probe-gene pairs  |  "
      f"{df_map['gene'].nunique():,} unique genes")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — INTEGRATE JETSET SCORES & SELECT BEST PROBE PER GENE
# ══════════════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print("  STEP 3 — Merge JetSet scores, select best probe per gene")
print("─" * 60)

if not os.path.exists(JETSET_CSV):
    print(f"CRITICAL: JetSet CSV not found at:\n  {JETSET_CSV}")
    sys.exit(1)

df_jetset = pd.read_csv(JETSET_CSV).rename(
    columns={"probeset": "probe", "overall": "jetset_score"}
)

df_map = pd.merge(
    df_map,
    df_jetset[["probe", "jetset_score"]],
    on   = "probe",
    how  = "left",
)
df_map["jetset_score"] = df_map["jetset_score"].fillna(-999)

# For each gene: highest JetSet score wins; intensity breaks ties
jetset_probe_df = (
    df_map
    .sort_values(["jetset_score", "intensity"], ascending=[False, False])
    .drop_duplicates(subset="gene", keep="first")
    .sort_values("gene")
    .reset_index(drop=True)
)

genes_unique   = jetset_probe_df["gene"].values          # final gene list
indices_jetset = jetset_probe_df["index"].values.astype(int)  # column indices in H5

print(f"  Final gene set : {len(genes_unique):,} unique genes")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — WRITE OUTPUT H5 (samples x genes)
# ══════════════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print("  STEP 4 — Build output H5")
print("─" * 60)

n_genes_final = len(genes_unique)

with h5py.File(INPUT_H5, "r") as f_in, h5py.File(OUTPUT_H5, "w") as f_out:

    dset_in  = f_in[KEY_EXPR]                      # (n_samples, n_probes)
    n_samples = dset_in.shape[0]

    # ── expressions : (n_samples, n_genes) ───────────────────────────────────
    dset_out = f_out.create_dataset(
        KEY_EXPR,
        shape            = (n_samples, n_genes_final),
        dtype            = "float32",
        chunks           = (min(500, n_samples), min(1_000, n_genes_final)),
        compression      = "gzip",
        compression_opts = 4,
    )

    for i in tqdm(range(0, n_samples, CHUNK_SIZE), desc="  Writing chunks"):
        end_i           = min(i + CHUNK_SIZE, n_samples)
        raw_chunk       = dset_in[i:end_i, :]              # (chunk, n_probes)
        dset_out[i:end_i, :] = raw_chunk[:, indices_jetset] # (chunk, n_genes)

    # ── gsm_ids : copied verbatim ─────────────────────────────────────────────
    gsm_raw = f_in[KEY_GSM][:]
    f_out.create_dataset(KEY_GSM, data=gsm_raw)

    # ── gene_names ────────────────────────────────────────────────────────────
    f_out.create_dataset(
        "gene_names",
        data=np.array(genes_unique, dtype="S"),   # byte strings, same as probe_ids
    )

size_mb = os.path.getsize(OUTPUT_H5) / 1024 ** 2
print(f"\n  Output written: {OUTPUT_H5}  ({size_mb:.1f} MB)")
print(f"  Shape: ({n_samples:,} samples  x  {n_genes_final:,} genes)")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SANITY CHECK & DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════════════════════════

print("\n─" * 60)
print("  STEP 5 — Sanity check & distribution plot")
print("─" * 60)

with h5py.File(OUTPUT_H5, "r") as f:
    # Verify output structure mirrors expected schema
    for key in (KEY_EXPR, KEY_GSM, "gene_names"):
        assert key in f, f"Missing dataset '{key}' in output H5"

    expr_shape = f[KEY_EXPR].shape
    print(f"  expressions : {expr_shape[0]:,} x {expr_shape[1]:,}  ✓")
    print(f"  gsm_ids     : {f[KEY_GSM].shape[0]:,}  ✓")
    print(f"  gene_names  : {f['gene_names'].shape[0]:,}  ✓")

    # Load a small subset for the distribution plot
    sample_data = f[KEY_EXPR][:500, :1_000].flatten()

print(f"\n  Intensity stats (500 samples x 1000 genes subset):")
print(f"    Mean   : {np.mean(sample_data):.4f}")
print(f"    Std    : {np.std(sample_data):.4f}")
print(f"    Min/Max: {np.min(sample_data):.4f} / {np.max(sample_data):.4f}")
print(f"    NaNs   : {np.isnan(sample_data).sum()}")

plt.figure(figsize=(10, 5))
sns.kdeplot(sample_data, fill=True, color="forestgreen", alpha=0.5,
            label="JetSet best-probe per gene")
plt.title("Gene Expression Distribution (JetSet Probe Selection)")
plt.xlabel("Intensity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

print("\n  Done.")  
