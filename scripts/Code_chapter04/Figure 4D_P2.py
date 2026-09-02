"""
=============================================================
Residual Proteome Clustering - DRP Independence Validation
=============================================================
Objectives:
  1. Regress out efficacy/toxicity/dose/technical variables.
  2. Re-cluster using the residual proteome.
  3. Compare Original DRP vs. Residual DRP.
  4. Validate if MOA enrichments are preserved.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from umap import UMAP
import leidenalg
import igraph as ig
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set

from collections import Counter, OrderedDict
from scipy.stats import fisher_exact
import warnings
from contextlib import suppress
import os
import time
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter04"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter04"
CHAPTER03_RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter03"
OUTPUT_DIR = RESULTS_ROOT
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
})


def save_fig(name, dpi=300):
    """Save figures in both PDF and PNG formats"""
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', format='pdf')
    plt.savefig(path + ".png", dpi=dpi, bbox_inches='tight', format='png')
    print(f"  ✓ Saved: {name}")


# =============================================================
#  Helper Functions (Leiden Clustering)
# =============================================================
def _get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists,
                                                   n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)
    for i in range(n_obs):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue
            idx = i * n_neighbors + j
            rows[idx] = i
            cols[idx] = knn_indices[i, j]
            vals[idx] = knn_dists[i, j]
    return csr_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))


def get_igraph_from_adjacency(adjacency, directed=False):
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets)))
    with suppress(KeyError):
        g.es["weight"] = weights
    return g


def _compute_connectivities_umap(knn_indices, knn_dists, n_obs, n_neighbors,
                                 set_op_mix_ratio=1.0, local_connectivity=1.0):
    from scipy.sparse import coo_matrix
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
        connectivities = fuzzy_simplicial_set(
            X, n_neighbors, None, None,
            knn_indices=knn_indices, knn_dists=knn_dists,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity)
        if isinstance(connectivities, tuple):
            connectivities = connectivities[0]
        distances = _get_sparse_matrix_from_indices_distances_umap(
            knn_indices, knn_dists, n_obs, n_neighbors)
        return distances, connectivities.tocsr()


def leiden_cluster(X_std, resolution=0.88, seed=42):
    """Leiden Clustering"""
    pca = PCA(n_components=min(200, X_std.shape[0] - 1, X_std.shape[1]))
    pcs = pca.fit_transform(X_std)
    knn_indices, knn_dists, _ = nearest_neighbors(
        pcs, n_neighbors=20, metric='cosine',
        metric_kwds={}, angular=False, random_state=42, verbose=False)
    _, connectivities = _compute_connectivities_umap(
        knn_indices, knn_dists, pcs.shape[0], 20)
    G = get_igraph_from_adjacency(connectivities, directed=True)
    weights = np.array(G.es["weight"]).astype(np.float64)
    partition = leidenalg.find_partition(
        G, leidenalg.RBConfigurationVertexPartition,
        weights=weights, resolution_parameter=resolution,
        n_iterations=-1, seed=seed)

    # ★ Force clustering labels to start from 1 instead of 0
    return np.array(partition.membership) + 1


# =============================================================
#  STEP 1: Compute Residual Proteome
# =============================================================
def step1_compute_residuals(data, metadata):
    """
    Core: Regress out the following variables for each protein:
      1. Euclidean_Dist (Global perturbation magnitude)
      2. Growth inhibition (Cytotoxicity)
      3. log10(concentration) (Dose)
      4. Proteome depth (Number of non-missing proteins)
      5. Missingness (Proportion of missing values)

    Note:
      - Use **Linear Regression** to remove these effects and keep residuals.
      - Do not over-correct (do not include DRPs themselves).
    """
    print("\n" + "=" * 70)
    print("  STEP 1: COMPUTING RESIDUAL PROTEOME")
    print("=" * 70)

    # ---------- Prepare covariates ----------
    covariates = pd.DataFrame(index=data.index)

    # 1. Euclidean distance
    covariates['Euclidean_Dist'] = metadata['Euclidean_Dist'].fillna(
        metadata['Euclidean_Dist'].median()
    )

    # 2. Growth inhibition (handle missing values)
    gi = metadata['Growth inhibition (%) at the tested concentration'].copy()
    gi = gi.replace('not available', np.nan)
    gi = pd.to_numeric(gi, errors='coerce')
    covariates['Growth_Inhibition'] = gi.fillna(gi.median())

    # 3. log10(concentration)
    conc = metadata['Omics Dosing Concentration (μM, 24 h)'].copy()
    covariates['Log10_Conc'] = np.log10(conc.clip(lower=0.01))

    # 4. Proteome depth (Number of non-missing proteins per sample)
    covariates['Proteome_Depth'] = data.notna().sum(axis=1)

    # 5. Missingness (Proportion of missing values)
    covariates['Missingness'] = data.isna().sum(axis=1) / data.shape[1]

    # Standardize covariates
    scaler_cov = StandardScaler()
    X_cov = scaler_cov.fit_transform(covariates.values)

    print(f"\n  Covariates summary:")
    print(covariates.describe().round(3))

    # ---------- Regress out for each protein ----------
    n_proteins = data.shape[1]
    residuals = np.zeros_like(data.values)
    r2_scores = []

    print(f"\n  Regressing out covariates from {n_proteins} proteins...")

    for j in range(n_proteins):
        if (j + 1) % 500 == 0:
            print(f"    {j + 1}/{n_proteins}...")

        y = data.iloc[:, j].values

        # Handle missing values (fill with mean)
        mask_valid = ~np.isnan(y)
        if mask_valid.sum() < 10:  # Too few valid values, skip
            residuals[:, j] = y
            r2_scores.append(0)
            continue

        y_filled = y.copy()
        y_filled[~mask_valid] = np.nanmean(y)

        # Linear regression
        lr = LinearRegression()
        lr.fit(X_cov, y_filled)

        # Calculate residuals
        y_pred = lr.predict(X_cov)
        res = y_filled - y_pred
        residuals[:, j] = res

        # R² (Variance explained by covariates)
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_filled - np.mean(y_filled)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    # ---------- Statistics ----------
    r2_scores = np.array(r2_scores)
    print(f"\n  Regression R² distribution:")
    print(f"    Mean:   {r2_scores.mean():.4f}")
    print(f"    Median: {np.median(r2_scores):.4f}")
    print(f"    Max:    {r2_scores.max():.4f}")
    print(f"    Proteins with R²>0.3: {(r2_scores > 0.3).sum()} "
          f"({(r2_scores > 0.3).sum() / n_proteins * 100:.1f}%)")

    # Visualize R² distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r2_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=r2_scores.mean(), color='red', linestyle='--',
               label=f'Mean R²={r2_scores.mean():.3f}')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5,
               label='R²=0.3 threshold')
    ax.set_xlabel('Regression R² (per protein)')
    ax.set_ylabel('Number of Proteins')
    ax.set_title('How much variance is explained by covariates?\n'
                 '(Higher R² = more influenced by efficacy/toxicity/dose)')
    ax.legend()
    plt.tight_layout()
    save_fig("01_regression_r2_distribution")
    plt.close()

    # ---------- Construct Residuals DataFrame ----------
    df_residuals = pd.DataFrame(
        residuals,
        index=data.index,
        columns=data.columns
    )

    # Save
    df_residuals.to_csv(os.path.join(OUTPUT_DIR, "01_residual_proteome.csv"))
    covariates.to_csv(os.path.join(OUTPUT_DIR, "01_covariates_used.csv"))
    pd.DataFrame({
        'Protein': data.columns,
        'R2': r2_scores
    }).to_csv(os.path.join(OUTPUT_DIR, "01_protein_r2_scores.csv"), index=False)

    return df_residuals, covariates, r2_scores


# =============================================================
#  STEP 2: Residual DRP Clustering
# =============================================================
def step2_residual_clustering(df_residuals, resolution=0.88, n_seeds=50):
    """
    Re-run Leiden clustering on the residual proteome.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: CLUSTERING ON RESIDUAL PROTEOME")
    print("=" * 70)

    # Standardization (Residuals also need to be standardized)
    scaler = StandardScaler()
    X_res_std = scaler.fit_transform(df_residuals.values)

    # ===== 2A. Find best seed producing k=10 =====
    print(f"\n  Running Leiden with resolution={resolution} ({n_seeds} seeds)...")

    k_values = []
    all_labels = []

    for s in range(n_seeds):
        if (s + 1) % 10 == 0:
            print(f"    seed {s + 1}/{n_seeds}...")

        labels = leiden_cluster(X_res_std, resolution=resolution, seed=s)
        k = len(set(labels))
        k_values.append(k)
        all_labels.append(labels)

    k_dist = Counter(k_values)
    print(f"\n  k distribution: {dict(sorted(k_dist.items()))}")

    # Select seeds producing k=10
    k10_indices = [i for i, k in enumerate(k_values) if k == 10]
    print(f"  Seeds producing k=10: {len(k10_indices)}/{n_seeds}")

    if len(k10_indices) == 0:
        print("  ⚠ No seed produced k=10, using most common k instead")
        target_k = k_dist.most_common(1)[0][0]
        target_indices = [i for i, k in enumerate(k_values) if k == target_k]
    else:
        target_k = 10
        target_indices = k10_indices

    # Select the most representative one (highest mean ARI with other same-k results)
    if len(target_indices) == 1:
        best_idx = target_indices[0]
    else:
        mean_aris = []
        for i in target_indices:
            aris = [adjusted_rand_score(all_labels[i], all_labels[j])
                    for j in target_indices if j != i]
            mean_aris.append(np.mean(aris))
        best_idx = target_indices[np.argmax(mean_aris)]

    residual_labels = all_labels[best_idx]

    print(f"\n  Selected: k={len(set(residual_labels))}, seed={best_idx}")
    print(f"  Cluster sizes: {dict(sorted(Counter(residual_labels).items()))}")

    # ===== 2B. Compute Quality Metrics =====
    sil = silhouette_score(X_res_std, residual_labels, metric='cosine')
    print(f"  Silhouette: {sil:.4f}")

    # UMAP Visualization
    print("\n  Computing UMAP for residual space...")
    umap_res = UMAP(n_neighbors=20, min_dist=0.01, n_components=2,
                    metric='cosine', random_state=42, verbose=False
                    ).fit_transform(X_res_std)

    # Save
    df_res_labels = pd.DataFrame({
        'Drug': df_residuals.index,
        'Residual_Cluster': residual_labels
    })
    df_res_labels.to_csv(os.path.join(OUTPUT_DIR, "02_residual_drp_labels.csv"),
                         index=False)

    pd.DataFrame(umap_res, columns=['UMAP1', 'UMAP2'],
                 index=df_residuals.index
                 ).to_csv(os.path.join(OUTPUT_DIR, "02_residual_umap.csv"))

    return residual_labels, X_res_std, umap_res


# =============================================================
#  STEP 3: Compare Original DRP vs Residual DRP
# =============================================================
def step3_compare_drps(original_labels, residual_labels, drug_names,
                       umap_orig, umap_res):
    """
    Core Metrics:
      1. ARI / NMI (Numerical Consistency)
      2. Cluster Correspondence (Sankey Diagram)
      3. Per-cluster Retention Rate
    """
    print("\n" + "=" * 70)
    print("  STEP 3: COMPARING ORIGINAL vs RESIDUAL DRP")
    print("=" * 70)

    # ===== 3A. ARI / NMI =====
    ari = adjusted_rand_score(original_labels, residual_labels)
    nmi = normalized_mutual_info_score(original_labels, residual_labels)

    print(f"\n  {'=' * 50}")
    print(f"  ★ ARI (Adjusted Rand Index):      {ari:.4f}")
    print(f"  ★ NMI (Normalized Mutual Info):   {nmi:.4f}")
    print(f"  {'=' * 50}")

    # Judging criteria
    if ari >= 0.6:
        verdict = "✅ STRONG AGREEMENT - DRP is robust beyond efficacy"
    elif ari >= 0.4:
        verdict = "⚠ MODERATE AGREEMENT - Partial mechanism signal preserved"
    else:
        verdict = "❌ WEAK AGREEMENT - DRP may be driven by efficacy/toxicity"

    print(f"\n  Verdict: {verdict}")

    # ===== 3B. Cluster Correspondence Matrix =====
    ct = pd.crosstab(
        pd.Series(original_labels, name='Original_DRP'),
        pd.Series(residual_labels, name='Residual_DRP')
    )

    print(f"\n  Cluster correspondence (count):")
    print(ct.to_string())

    # Normalize (Row sum = 1, observing where original clusters flow to)
    ct_pct = ct.div(ct.sum(axis=1), axis=0)

    # Primary destination for each original cluster
    print(f"\n  Per-cluster retention:")
    for orig_c in sorted(set(original_labels)):
        if orig_c in ct_pct.index:
            row = ct_pct.loc[orig_c]
            top_res_c = row.idxmax()
            retention = row.max()
            print(f"    Original C{orig_c} → Residual C{top_res_c} "
                  f"({retention * 100:.1f}% retained)")

    # Save
    ct.to_csv(os.path.join(OUTPUT_DIR, "03_cluster_correspondence.csv"))
    ct_pct.to_csv(os.path.join(OUTPUT_DIR, "03_cluster_correspondence_pct.csv"))

    # ===== 3C. Visualization =====
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # (a) Correspondence heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(ct_pct, annot=ct.values, fmt='d', cmap='YlOrRd',
                vmin=0, vmax=1, ax=ax1, linewidths=0.5,
                cbar_kws={'label': 'Proportion', 'shrink': 0.7})
    ax1.set_title('(a) Cluster Correspondence\n'
                  f'(ARI={ari:.3f}, NMI={nmi:.3f})',
                  fontweight='bold')
    ax1.set_xlabel('Residual DRP')
    ax1.set_ylabel('Original DRP')

    # (b) UMAP - Original
    ax2 = fig.add_subplot(gs[0, 1])
    colors_orig = plt.cm.tab20(np.linspace(0, 1, len(set(original_labels))))
    for c_idx, c in enumerate(sorted(set(original_labels))):
        m = original_labels == c
        ax2.scatter(umap_orig[m, 0], umap_orig[m, 1],
                    c=[colors_orig[c_idx]], s=20, alpha=0.7,
                    edgecolors='white', linewidth=0.2,
                    label=f'C{c}({np.sum(m)})')
    ax2.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    ax2.set_title('(b) Original DRP (before regression)', fontweight='bold')

    # (c) UMAP - Residual
    ax3 = fig.add_subplot(gs[0, 2])
    colors_res = plt.cm.tab20(np.linspace(0, 1, len(set(residual_labels))))
    for c_idx, c in enumerate(sorted(set(residual_labels))):
        m = residual_labels == c
        ax3.scatter(umap_res[m, 0], umap_res[m, 1],
                    c=[colors_res[c_idx]], s=20, alpha=0.7,
                    edgecolors='white', linewidth=0.2,
                    label=f'C{c}({np.sum(m)})')
    ax3.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax3.set_xlabel('UMAP1')
    ax3.set_ylabel('UMAP2')
    ax3.set_title('(c) Residual DRP (after regression)', fontweight='bold')

    # (d) Sankey-like flow diagram
    ax4 = fig.add_subplot(gs[1, :])

    # Draw flow from Original to Residual
    orig_clusters = sorted(set(original_labels))
    res_clusters = sorted(set(residual_labels))

    y_orig = np.arange(len(orig_clusters))
    y_res = np.arange(len(res_clusters))

    # Left side: Original clusters
    for i, c in enumerate(orig_clusters):
        n = np.sum(original_labels == c)
        ax4.barh(i, 1, left=0, height=0.8,
                 color=colors_orig[i], alpha=0.7,
                 edgecolor='black', linewidth=1)
        ax4.text(-0.1, i, f'Orig C{c}\n(n={n})',
                 ha='right', va='center', fontsize=8, fontweight='bold')

    # Right side: Residual clusters
    for i, c in enumerate(res_clusters):
        n = np.sum(residual_labels == c)
        ax4.barh(i, 1, left=2, height=0.8,
                 color=colors_res[i], alpha=0.7,
                 edgecolor='black', linewidth=1)
        ax4.text(3.1, i, f'Res C{c}\n(n={n})',
                 ha='left', va='center', fontsize=8, fontweight='bold')

    # Draw connecting lines (only main flows, >10%)
    for i, orig_c in enumerate(orig_clusters):
        if orig_c not in ct_pct.index:
            continue
        for j, res_c in enumerate(res_clusters):
            if res_c not in ct_pct.columns:
                continue

            frac = ct_pct.loc[orig_c, res_c]
            if frac < 0.1:  # Ignore minor flows
                continue

            # Bezier curve simulation
            x = np.linspace(1, 2, 100)
            y_start = i
            y_end = j
            y_curve = y_start + (y_end - y_start) * ((x - 1) ** 2)

            ax4.plot(x, y_curve, color=colors_orig[i],
                     linewidth=frac * 20, alpha=0.4, zorder=1)

    ax4.set_xlim(-1, 4)
    ax4.set_ylim(-0.5, max(len(orig_clusters), len(res_clusters)) - 0.5)
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.set_title('(d) Cluster Flow: Original → Residual DRP\n'
                  '(line width ∝ drug count)',
                  fontweight='bold', pad=20)

    plt.suptitle(f'DRP Comparison: Original vs Residual\n'
                 f'ARI={ari:.3f}, NMI={nmi:.3f} — {verdict}',
                 fontsize=14, fontweight='bold', y=0.98)

    save_fig("03_drp_comparison")
    plt.close()

    return ari, nmi, ct, ct_pct


# =============================================================
#  STEP 4: MOA Enrichment Validation
# =============================================================
def step4_moa_enrichment(original_labels, residual_labels, metadata, df_moa):
    """
    Test: Are drugs with the same MOA still enriched in Residual DRPs?

    Parameters
    ----------
    original_labels : array
        Original DRP labels
    residual_labels : array
        Residual DRP labels
    metadata : DataFrame
        Metadata (Contains Compound_ID, etc.)
    df_moa : DataFrame
        MOA annotation table, index=Compound_ID, must contain 'MOA' and 'Target'
    """
    print("\n" + "=" * 70)
    print("  STEP 4: MOA ENRICHMENT IN RESIDUAL DRP")
    print("=" * 70)

    # ===== 4A. Align Data =====
    # Ensure metadata and df_moa share the same index
    common_ids = metadata.index.intersection(df_moa.index)
    print(f"\n  Common compounds: {len(common_ids)}/{len(metadata)}")

    if len(common_ids) == 0:
        print("  ⚠ No overlap between metadata and MOA table!")
        print(f"  Metadata index example: {metadata.index[:5].tolist()}")
        print(f"  MOA table index example: {df_moa.index[:5].tolist()}")
        return None, None

    # Extract MOA information
    df_moa_aligned = df_moa.loc[common_ids]

    # MOA distribution
    moa_counts = df_moa_aligned['MOA'].value_counts()
    print(f"\n  MOA distribution (top 15):")
    print(moa_counts.head(15).to_string())

    # ===== 4B. Construct MOA categories (keep n≥5) =====
    moa_categories = {}
    for moa in moa_counts.index:
        if pd.isna(moa) or moa == 'Unknown' or moa == '':
            continue
        drugs = df_moa_aligned[df_moa_aligned['MOA'] == moa].index.tolist()
        if len(drugs) >= 5:  # At least 5 drugs required
            moa_categories[moa] = drugs

    print(f"\n  MOA categories for enrichment: {len(moa_categories)}")
    print(f"  Examples: {list(moa_categories.keys())[:5]}")

    if len(moa_categories) == 0:
        print("  ⚠ No MOA categories with ≥5 drugs, skipping enrichment")
        return None, None

    # ===== 4C. Fisher's Exact Test =====
    print(f"\n  Running Fisher's exact test...")

    enrichment_results = []

    # Map drug_id to index in residual_labels
    drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(metadata.index)}

    for moa, moa_drugs in moa_categories.items():
        moa_indices = [drug_id_to_idx[d] for d in moa_drugs if d in drug_id_to_idx]
        if len(moa_indices) == 0:
            continue

        moa_mask = np.zeros(len(residual_labels), dtype=bool)
        moa_mask[moa_indices] = True
        n_moa = moa_mask.sum()

        for c in sorted(set(residual_labels)):
            cluster_mask = residual_labels == c
            n_cluster = cluster_mask.sum()

            # 2x2 Contingency Table
            a = np.sum(moa_mask & cluster_mask)       # In MOA & In Cluster
            b = np.sum(moa_mask & ~cluster_mask)      # In MOA & Not in Cluster
            c_val = np.sum(~moa_mask & cluster_mask)  # Not in MOA & In Cluster
            d = np.sum(~moa_mask & ~cluster_mask)     # Not in MOA & Not in Cluster

            # Expected Value
            expected = n_moa * n_cluster / len(residual_labels)

            # Fold Enrichment
            if expected > 0:
                enrichment = a / expected
            else:
                enrichment = np.nan

            # Fisher's exact test
            try:
                from scipy.stats import fisher_exact
                oddsratio, pval = fisher_exact([[a, b], [c_val, d]],
                                               alternative='greater')
            except:
                oddsratio, pval = np.nan, 1.0

            enrichment_results.append({
                'MOA': moa,
                'Residual_Cluster': c,
                'N_MOA_total': n_moa,
                'N_in_cluster': a,
                'N_cluster_total': n_cluster,
                'Expected': round(expected, 2),
                'Enrichment': round(enrichment, 2) if not np.isnan(enrichment) else np.nan,
                'OR': round(oddsratio, 2) if not np.isnan(oddsratio) else np.nan,
                'P_value': pval
            })

    df_enrich = pd.DataFrame(enrichment_results)

    # FDR Correction
    from statsmodels.stats.multitest import multipletests
    _, padj, _, _ = multipletests(df_enrich['P_value'].fillna(1).values,
                                  method='fdr_bh')
    df_enrich['P_adjusted'] = padj
    df_enrich['Significant'] = (df_enrich['P_adjusted'] < 0.05) & (df_enrich['N_in_cluster'] >= 3)

    df_enrich = df_enrich.sort_values('P_adjusted')

    print(f"\n  Significant MOA enrichments (FDR < 0.05, n≥3):")
    sig = df_enrich[df_enrich['Significant']]
    if len(sig) > 0:
        print(sig[['MOA', 'Residual_Cluster', 'N_in_cluster', 'N_MOA_total',
                   'Enrichment', 'P_adjusted']].head(20).to_string(index=False))
    else:
        print("    None found")

    # Save
    df_enrich.to_csv(os.path.join(OUTPUT_DIR, "04_moa_enrichment_residual.csv"),
                     index=False)

    # ===== 4D. Visualization =====
    df_plot = df_enrich[df_enrich['Significant']].copy()

    if len(df_plot) > 0:
        # Limit to top 30
        df_plot = df_plot.head(30)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.3)))

        # Calculate -log10(FDR)
        neg_log_fdr = -np.log10(df_plot['P_adjusted'].clip(lower=1e-10))

        scatter = ax.scatter(
            df_plot['Enrichment'],
            range(len(df_plot)),
            s=df_plot['N_in_cluster'] * 50 + 30,
            c=neg_log_fdr,
            cmap='Reds', vmin=0, vmax=max(5, neg_log_fdr.max()),
            edgecolors='black', linewidth=0.5, alpha=0.85
        )

        ax.set_yticks(range(len(df_plot)))
        labels = [f"{row['MOA'][:40]} → ResC{row['Residual_Cluster']} "
                  f"({row['N_in_cluster']}/{row['N_MOA_total']})"
                  for _, row in df_plot.iterrows()]
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Fold Enrichment (Observed / Expected)', fontsize=11)
        ax.set_title('MOA Enrichment in Residual DRP\n'
                     '(After regressing out efficacy/toxicity signals)\n'
                     'Size = N drugs in cluster, Color = -log₁₀(FDR)',
                     fontweight='bold', fontsize=12)
        ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.invert_yaxis()

        # Highlight highly significant enrichments (Enrichment > 2)
        for i, (_, row) in enumerate(df_plot.iterrows()):
            if row['Enrichment'] > 2:
                ax.scatter(row['Enrichment'], i, s=200,
                           facecolors='none', edgecolors='red',
                           linewidths=2, zorder=10)

        plt.colorbar(scatter, ax=ax, label='-log₁₀(FDR)', shrink=0.5)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='grey', markersize=8,
                   label='3 drugs'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='grey', markersize=12,
                   label='5 drugs'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='grey', markersize=16,
                   label='10 drugs'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=9, title='N drugs')

        plt.tight_layout()
        save_fig("04_moa_enrichment_residual")
        plt.close()

        print(f"\n  ✓ Saved enrichment plot ({len(df_plot)} MOA-cluster pairs)")
    else:
        print(f"\n  ⚠ No significant enrichments to plot")

    # ===== 4E. Compare MOA Enrichment: Original vs Residual =====
    print(f"\n  Comparing MOA enrichment: Original vs Residual DRP...")

    enrichment_orig = []
    for moa, moa_drugs in moa_categories.items():
        moa_indices = [drug_id_to_idx[d] for d in moa_drugs if d in drug_id_to_idx]
        if len(moa_indices) == 0:
            continue

        moa_mask = np.zeros(len(original_labels), dtype=bool)
        moa_mask[moa_indices] = True
        n_moa = moa_mask.sum()

        for c in sorted(set(original_labels)):
            cluster_mask = original_labels == c
            n_cluster = cluster_mask.sum()

            a = np.sum(moa_mask & cluster_mask)
            b = np.sum(moa_mask & ~cluster_mask)
            c_val = np.sum(~moa_mask & cluster_mask)
            d = np.sum(~moa_mask & ~cluster_mask)

            expected = n_moa * n_cluster / len(original_labels)
            enrichment = a / expected if expected > 0 else np.nan

            try:
                oddsratio, pval = fisher_exact([[a, b], [c_val, d]],
                                               alternative='greater')
            except:
                oddsratio, pval = np.nan, 1.0

            enrichment_orig.append({
                'MOA': moa,
                'Original_Cluster': c,
                'N_in_cluster': a,
                'Expected': round(expected, 2),
                'Enrichment': round(enrichment, 2) if not np.isnan(enrichment) else np.nan,
                'P_value': pval
            })

    df_enrich_orig = pd.DataFrame(enrichment_orig)
    _, padj_orig, _, _ = multipletests(df_enrich_orig['P_value'].fillna(1).values,
                                       method='fdr_bh')
    df_enrich_orig['P_adjusted'] = padj_orig
    df_enrich_orig['Significant'] = (df_enrich_orig['P_adjusted'] < 0.05) & \
                                    (df_enrich_orig['N_in_cluster'] >= 3)

    df_enrich_orig.to_csv(os.path.join(OUTPUT_DIR, "04_moa_enrichment_original.csv"),
                          index=False)

    n_sig_orig = df_enrich_orig['Significant'].sum()
    n_sig_res = df_enrich['Significant'].sum()

    print(f"  Significant enrichments:")
    print(f"    Original DRP: {n_sig_orig}")
    print(f"    Residual DRP: {n_sig_res}")

    if n_sig_res >= n_sig_orig * 0.7:
        print(f"  ✅ Residual DRP retains {n_sig_res / max(n_sig_orig, 1) * 100:.0f}% "
              f"of original MOA enrichments")
        print(f"     → Mechanism signal is preserved!")
    else:
        print(f"  ⚠ Residual DRP only retains {n_sig_res / max(n_sig_orig, 1) * 100:.0f}% "
              f"of original MOA enrichments")

    return df_enrich, df_enrich_orig

# =============================================================
#  STEP 5: External Dataset Validation
# =============================================================
def step5_external_validation(X_res_std, residual_labels,
                              external_data_path=None):
    """
    Optional: Perform residual regression on external datasets to test conservation.
    """
    print("\n" + "=" * 70)
    print("  STEP 5: EXTERNAL DATASET VALIDATION (Optional)")
    print("=" * 70)

    if external_data_path is None:
        print("  ⚠ No external data provided, skipping")
        return None

    print("  (Implementation depends on external data format)")
    return None

# =============================================================
#                        ★ MAIN ★
# =============================================================
if __name__ == "__main__":
    T0 = time.time()
    print("=" * 70)
    print("  RESIDUAL DRP VALIDATION PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ========== Read Data ==========
    # Original Proteome
    data = pd.read_csv(
        DATA_ROOT / "14_367_Drugs_log2FC_6682_for_UMAP.csv",
        delimiter=',', index_col=0
    )
    print(f"\nData: {data.shape[0]} drugs × {data.shape[1]} proteins")

    # Metadata (Contains efficacy, dose, etc.)
    metadata = pd.read_csv(
        DATA_ROOT / "Table S2-DRP_inhibition-all.csv",
        index_col='Compound_ID'  # Assuming this is the drug name column
    )

    # Align (Ensure ordering is consistent)
    common_drugs = data.index.intersection(metadata.index)
    data = data.loc[common_drugs]
    metadata = metadata.loc[common_drugs]

    print(f"Aligned: {len(common_drugs)} drugs")

    # Read original DRP labels
    df_orig_labels = pd.read_csv(
        CHAPTER03_RESULTS_ROOT / "01_HepG2_k10_clustering" / "01_final_k10_labels.csv"
    )
    df_orig_labels = df_orig_labels.set_index('Drug')
    original_labels = df_orig_labels.loc[data.index, 'Cluster'].values

    # Read original UMAP
    df_orig_umap = pd.read_csv(
        CHAPTER03_RESULTS_ROOT / "01_HepG2_k10_clustering" / "07_umap_coordinates.csv",
        index_col='Drug'
    )
    umap_orig = df_orig_umap.loc[data.index, ['UMAP1', 'UMAP2']].values

    print(f"Original DRP: {dict(sorted(Counter(original_labels).items()))}")

    # ========== STEP 1: Compute Residuals ==========
    df_residuals, covariates, r2_scores = step1_compute_residuals(
        data, metadata
    )

    # ========== STEP 2: Residual Clustering ==========
    residual_labels, X_res_std, umap_res = step2_residual_clustering(
        df_residuals, resolution=0.88, n_seeds=50
    )

    # ========== STEP 3: Compare DRPs ==========
    ari, nmi, ct, ct_pct = step3_compare_drps(
        original_labels, residual_labels, data.index.tolist(),
        umap_orig, umap_res
    )

    # ========== STEP 4: MOA Enrichment ==========
    # Read MOA annotations
    df_moa = pd.read_csv(
        DATA_ROOT / "Table S2-DRP_inhibition-all.csv",
        index_col='Compound_ID'
    )

    print(f"\nMOA table loaded: {df_moa.shape[0]} compounds")
    print(f"Columns: {df_moa.columns.tolist()}")
    print(f"MOA examples:\n{df_moa[['MOA', 'Target']].head()}")

    df_enrich, df_enrich_orig = step4_moa_enrichment(
        original_labels, residual_labels, metadata, df_moa
    )

    # ========== STEP 5: External Validation (Optional) ==========
    # step5_external_validation(X_res_std, residual_labels)

    # ========== Done ==========
    elapsed = time.time() - T0
    print(f"\n{'=' * 70}")
    print(f"  ✅ DONE! Total time: {elapsed / 60:.1f} minutes")
    print(f"  📁 Outputs: {OUTPUT_DIR}")
    print(f"")
    print(f"  Key outputs:")
    print(f"    01_residual_proteome.csv")
    print(f"    02_residual_drp_labels.csv")
    print(f"    03_drp_comparison.pdf  ← Core Plot")
    print(f"    04_moa_enrichment_residual.csv")
    print(f"{'=' * 70}")
