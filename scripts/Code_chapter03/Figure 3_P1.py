#%%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import KMeans, SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix, csr_matrix
from umap import UMAP
import leidenalg
import igraph as ig
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set
from collections import Counter
import warnings
from contextlib import suppress
import os
import time
from datetime import datetime
from pathlib import Path

np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter03"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter03"
OUTPUT_DIR = RESULTS_ROOT / "01_HepG2_k10_clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig_name, dpi=600):
    path = os.path.join(OUTPUT_DIR, fig_name)
    plt.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', format='pdf')
    plt.savefig(path + ".png", dpi=dpi, bbox_inches='tight', format='png')
    print(f"  Saved: {fig_name}.pdf + .png")

TARGET_K = 10


# =============================================================
#                      Helper Functions
# =============================================================
def _get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X, n_neighbors, None, None,
        knn_indices=knn_indices, knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio, local_connectivity=local_connectivity)
    if isinstance(connectivities, tuple):
        connectivities = connectivities[0]
    distances = _get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n_obs, n_neighbors)
    return distances, connectivities.tocsr()


def leiden_cluster(X_std, resolution=1.0, seed=0):
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
    return np.array(partition.membership)


# =============================================================
#  STEP 1: Find the resolution that most frequently produces k=10
# =============================================================
def find_best_resolution_for_k10(X_std, n_seeds=50):
    """
    Finely scan resolution to find the one that most frequently produces k=10
    """
    print("\n" + "=" * 70)
    print(f"  STEP 1: FIND BEST RESOLUTION FOR K={TARGET_K}")
    print("=" * 70)

    # Fine scan
    resolutions = np.arange(0.75, 1.05, 0.01)
    results = []

    for res in resolutions:
        k_values = []
        for s in range(n_seeds):
            labels = leiden_cluster(X_std, resolution=res, seed=s)
            k_values.append(len(set(labels)))

        n_target = k_values.count(TARGET_K)
        pct_target = n_target / n_seeds * 100

        results.append({
            'resolution': round(res, 3),
            'k_mode': Counter(k_values).most_common(1)[0][0],
            'n_target_k': n_target,
            'pct_target_k': pct_target,
            'k_distribution': dict(Counter(k_values))
        })

        print(f"  res={res:.2f} → k={TARGET_K} in {n_target}/{n_seeds} seeds "
              f"({pct_target:.0f}%), k_mode={Counter(k_values).most_common(1)[0][0]}, "
              f"dist={dict(Counter(k_values))}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "00_resolution_scan_for_k10.csv"), index=False)

    # Select the resolution with the highest proportion of k=10
    best_idx = df_res['pct_target_k'].idxmax()
    best_res = df_res.loc[best_idx, 'resolution']
    best_pct = df_res.loc[best_idx, 'pct_target_k']

    print(f"\n  {'=' * 50}")
    print(f"  ★ Best resolution for k={TARGET_K}: {best_res}")
    print(f"  ★ k={TARGET_K} occurrence: {best_pct:.0f}% of seeds")
    print(f"  {'=' * 50}")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df_res['resolution'], df_res['pct_target_k'],
           width=0.008, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=best_res, color='red', linestyle='--', linewidth=2,
               label=f'Best: res={best_res} ({best_pct:.0f}%)')
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_ylabel(f'% of seeds producing k={TARGET_K}', fontsize=12)
    ax.set_title(f'Resolution Scan: Finding Best Resolution for k={TARGET_K}\n'
                 f'({n_seeds} seeds per resolution)', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "00_resolution_scan.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    return best_res, df_res


# =============================================================
#  STEP 2: Select the most representative k=10 labels at the best resolution
# =============================================================
def select_best_k10_labels(X_std, resolution, n_seeds=200):
    """
    Run many seeds, retain only those producing k=10, and select the most representative one
    """
    print("\n" + "=" * 70)
    print(f"  STEP 2: SELECT BEST K={TARGET_K} LABELS (res={resolution}, {n_seeds} seeds)")
    print("=" * 70)

    valid_labels = []   # Retain only k=TARGET_K
    valid_seeds = []

    for s in range(n_seeds):
        labels = leiden_cluster(X_std, resolution=resolution, seed=s)
        k = len(set(labels))
        if k == TARGET_K:
            valid_labels.append(labels)
            valid_seeds.append(s)

    print(f"  Seeds producing k={TARGET_K}: {len(valid_labels)}/{n_seeds} "
          f"({len(valid_labels)/n_seeds*100:.1f}%)")

    if len(valid_labels) == 0:
        raise ValueError(f"No seed produced k={TARGET_K} at resolution={resolution}! "
                         f"Try adjusting resolution.")

    if len(valid_labels) == 1:
        print(f"  Only 1 valid seed (seed={valid_seeds[0]}), using it directly.")
        return valid_labels[0], valid_seeds[0]

    # Calculate ARI among valid labels and select the most representative one
    mean_aris = []
    for i in range(len(valid_labels)):
        aris = [adjusted_rand_score(valid_labels[i], valid_labels[j])
                for j in range(len(valid_labels)) if j != i]
        mean_aris.append(np.mean(aris))

    best_idx = np.argmax(mean_aris)
    best_seed = valid_seeds[best_idx]
    ref_labels = valid_labels[best_idx]
    ref_labels = ref_labels + 1

    print(f"  Best seed: {best_seed} (mean ARI with other k={TARGET_K} runs = {mean_aris[best_idx]:.4f})")
    print(f"  Cluster sizes: {dict(sorted(Counter(ref_labels).items()))}")

    # Check cluster size distribution
    sizes = sorted(Counter(ref_labels).values(), reverse=True)
    print(f"  Size distribution: {sizes}")
    print(f"  Largest/Smallest ratio: {max(sizes)/max(min(sizes),1):.1f}x")

    # Save
    df = pd.DataFrame({
        'Drug': drug_names,
        'Cluster': ref_labels
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "01_final_k10_labels.csv"), index=False)

    return ref_labels, best_seed


# =============================================================
#  STEP 3: Bootstrap stability validation
# =============================================================
def bootstrap_stability(X_std, resolution, ref_labels, n_bootstrap=300):
    print("\n" + "=" * 70)
    print(f"  STEP 3: BOOTSTRAP STABILITY (n={n_bootstrap})")
    print("=" * 70)

    n_drugs = X_std.shape[0]
    n_sample = int(n_drugs * 0.8)
    n_ref = len(set(ref_labels))

    co_count = np.zeros((n_drugs, n_drugs), dtype=np.float64)
    co_appear = np.zeros((n_drugs, n_drugs), dtype=np.float64)
    boot_k = []

    t0 = time.time()
    for b in range(n_bootstrap):
        if (b + 1) % 30 == 0:
            el = time.time() - t0
            eta = el / (b + 1) * (n_bootstrap - b - 1)
            print(f"    {b+1}/{n_bootstrap} ({el:.0f}s elapsed, ~{eta:.0f}s remaining)")

        idx = np.random.choice(n_drugs, size=n_sample, replace=False)
        try:
            bl = leiden_cluster(X_std[idx], resolution=resolution, seed=b)
            boot_k.append(len(set(bl)))
            for ip in range(len(idx)):
                for jp in range(ip + 1, len(idx)):
                    di, dj = idx[ip], idx[jp]
                    co_appear[di, dj] += 1
                    co_appear[dj, di] += 1
                    if bl[ip] == bl[jp]:
                        co_count[di, dj] += 1
                        co_count[dj, di] += 1
        except:
            continue

    with np.errstate(divide='ignore', invalid='ignore'):
        co_prob = np.where(co_appear > 0, co_count / co_appear, 0)
    np.fill_diagonal(co_prob, 1.0)

    # Per-drug stability
    drug_stab = np.zeros(n_drugs)
    for i in range(n_drugs):
        same = np.where(ref_labels == ref_labels[i])[0]
        same = same[same != i]
        drug_stab[i] = np.mean(co_prob[i, same]) if len(same) > 0 else 1.0

    n_core = np.sum(drug_stab >= 0.7)
    print(f"\n  Mean stability:     {np.mean(drug_stab):.4f}")
    print(f"  Core drugs (≥0.7):  {n_core}/{n_drugs} ({n_core/n_drugs*100:.1f}%)")
    print(f"  Boot k mode:        {Counter(boot_k).most_common(1)[0]}")
    print(f"  Boot k range:       {min(boot_k)}-{max(boot_k)}")
    print(f"  Boot k distribution: {dict(sorted(Counter(boot_k).items()))}")

    # ===== Co-clustering heatmap =====
    sort_idx = np.argsort(ref_labels)
    co_sorted = co_prob[sort_idx][:, sort_idx]
    sorted_lab = ref_labels[sort_idx]

    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#4393C3', '#B2182B'])
    im = ax.imshow(co_sorted, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for c in range(1,n_ref+1):
        ci = np.where(sorted_lab == c)[0]
        if len(ci) > 0:
            bnd = ci[-1] + 0.5
            ax.axhline(y=bnd, color='black', linewidth=0.5)
            ax.axvline(x=bnd, color='black', linewidth=0.5)
            mid = ci[0] + len(ci) / 2
            ax.text(-10, mid, f'C{c}', fontsize=9, fontweight='bold', va='center')
    plt.colorbar(im, ax=ax, label='Co-clustering Probability', shrink=0.8)
    ax.set_title(f'Co-clustering Probability Matrix\n'
                 f'(k={n_ref}, resolution={resolution}, {n_bootstrap} bootstraps)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_co_clustering_heatmap.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    # ===== Bootstrap summary =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) k distribution
    ax = axes[0]
    kc = Counter(boot_k)
    ks = sorted(kc.keys())
    cols_bar = ['#B2182B' if k == TARGET_K else 'steelblue' for k in ks]
    ax.bar(ks, [kc[k] for k in ks], color=cols_bar, edgecolor='black')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Frequency')
    ax.set_title(f'(a) Bootstrap k Distribution (target={TARGET_K})')

    # (b) Drug stability distribution
    ax = axes[1]
    ax.hist(drug_stab, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(x=0.7, color='red', linestyle='--', label=f'Core≥0.7')
    ax.axvline(x=np.mean(drug_stab), color='blue', linestyle='--',
               label=f'Mean={np.mean(drug_stab):.3f}')
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Number of Drugs')
    ax.set_title(f'(b) Drug Stability (Core: {n_core/n_drugs*100:.1f}%)')
    ax.legend()

    # (c) Per-cluster stability
    ax = axes[2]
    clusters = sorted(set(ref_labels))
    c_stab = [np.mean(drug_stab[ref_labels == c]) for c in clusters]
    c_cols = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
    ax.bar(clusters, c_stab, color=c_cols, edgecolor='black')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean Stability')
    ax.set_title('(c) Per-Cluster Stability')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_bootstrap_summary.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    np.save(os.path.join(OUTPUT_DIR, "co_clustering_matrix.npy"), co_prob)

    return co_prob, drug_stab, boot_k


# =============================================================
#  STEP 4: Multi-method consistency
# =============================================================
def multi_method(X_std, ref_labels):
    print("\n" + "=" * 70)
    print("  STEP 4: MULTI-METHOD COMPARISON")
    print("=" * 70)

    k = len(set(ref_labels))
    methods = {'Leiden': ref_labels}

    try:
        Z = linkage(pdist(X_std, 'euclidean'), method='ward')
        methods['HClust_ward'] = fcluster(Z, t=k, criterion='maxclust')
    except Exception as e:
        print(f"  HClust_ward failed: {e}")
    try:
        Z = linkage(pdist(X_std, 'cosine'), method='complete')
        methods['HClust_complete'] = fcluster(Z, t=k, criterion='maxclust')
    except Exception as e:
        print(f"  HClust_complete failed: {e}")
    try:
        Z = linkage(pdist(X_std, 'cosine'), method='average')
        methods['HClust_average'] = fcluster(Z, t=k, criterion='maxclust')
    except Exception as e:
        print(f"  HClust_average failed: {e}")
    try:
        methods['KMeans'] = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_std)+1
    except Exception as e:
        print(f"  KMeans failed: {e}")
    try:
        methods['Spectral'] = SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors',
            n_neighbors=20, random_state=42).fit_predict(X_std)+1
    except Exception as e:
        print(f"  Spectral failed: {e}")

    names = list(methods.keys())
    n = len(names)
    ari_mat = np.zeros((n, n))
    nmi_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ari_mat[i, j] = adjusted_rand_score(methods[names[i]], methods[names[j]])
            nmi_mat[i, j] = normalized_mutual_info_score(methods[names[i]], methods[names[j]])

    ari_df = pd.DataFrame(ari_mat, index=names, columns=names)
    nmi_df = pd.DataFrame(nmi_mat, index=names, columns=names)

    print("\n  ARI matrix:")
    print(ari_df.round(4).to_string())

    leiden_vs = [ari_mat[0, j] for j in range(1, n)]
    print(f"\n  Leiden vs others mean ARI: {np.mean(leiden_vs):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(ari_df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                ax=axes[0], square=True)
    axes[0].set_title('(a) ARI Between Methods')
    sns.heatmap(nmi_df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                ax=axes[1], square=True)
    axes[1].set_title('(b) NMI Between Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_multi_method.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    ari_df.to_csv(os.path.join(OUTPUT_DIR, "04_ari_matrix.csv"))
    nmi_df.to_csv(os.path.join(OUTPUT_DIR, "04_nmi_matrix.csv"))

    return methods, ari_df, nmi_df


# =============================================================
#  STEP 5: Per-cluster + per-drug analysis
# =============================================================
def cluster_drug_analysis(X_std, ref_labels, drug_stab, drug_names):
    print("\n" + "=" * 70)
    print("  STEP 5: PER-CLUSTER & PER-DRUG ANALYSIS")
    print("=" * 70)

    clusters = sorted(set(ref_labels))
    n_clusters = len(clusters)
    sil_s = silhouette_samples(X_std, ref_labels, metric='cosine')

    stats = []
    for c in clusters:
        m = ref_labels == c
        n = int(np.sum(m))
        mean_sil = float(np.mean(sil_s[m]))
        mean_stab = float(np.mean(drug_stab[m]))
        n_core = int(np.sum(drug_stab[m] >= 0.7))

        stats.append({
            'Cluster': c,
            'N_drugs': n,
            'Pct': f"{n / len(ref_labels) * 100:.1f}%",
            'Silhouette': round(mean_sil, 4),
            'Stability': round(mean_stab, 4),
            'N_core': n_core,
            'Core_pct': f"{n_core / n * 100:.0f}%",
            'Quality': 'High' if (mean_sil > 0 and mean_stab > 0.6) else
                       'Medium' if mean_stab > 0.5 else 'Low'
        })

    df_stats = pd.DataFrame(stats)
    print("\n" + df_stats.to_string(index=False))
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "05_cluster_quality.csv"), index=False)

    # Per-drug report
    df_drug = pd.DataFrame({
        'Drug': drug_names,
        'Cluster': ref_labels,
        'Silhouette': np.round(sil_s, 4),
        'Stability': np.round(drug_stab, 4),
        'Category': ['Core' if s >= 0.7 else 'Boundary' for s in drug_stab]
    }).sort_values(['Cluster', 'Stability'], ascending=[True, False])
    df_drug.to_csv(os.path.join(OUTPUT_DIR, "06_drug_report.csv"), index=False)

    # ========== Visualization ==========
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # (a) Silhouette
    ax = axes[0]
    y_lower = 0
    for idx, c in enumerate(clusters):
        vals = np.sort(sil_s[ref_labels == c])
        y_upper = y_lower + len(vals)
        ax.barh(range(y_lower, y_upper), vals, height=1.0,
                color=colors[idx], edgecolor='none', alpha=0.8)
        ax.text(-0.08, y_lower + len(vals) / 2, f'C{c}',
                fontsize=8, fontweight='bold', va='center')
        y_lower = y_upper + 2
    ax.axvline(x=np.mean(sil_s), color='red', linestyle='--',
               label=f'Mean={np.mean(sil_s):.3f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_title('(a) Silhouette Plot')
    ax.legend()

    # (b) Quality map
    ax = axes[1]
    for idx, c in enumerate(clusters):
        ax.scatter(df_stats.loc[idx, 'Silhouette'], df_stats.loc[idx, 'Stability'],
                   s=df_stats.loc[idx, 'N_drugs'] * 8, c=[colors[idx]],
                   edgecolors='black', alpha=0.8, zorder=3)
        ax.annotate(f"C{c}\n(n={df_stats.loc[idx, 'N_drugs']})",
                    (df_stats.loc[idx, 'Silhouette'], df_stats.loc[idx, 'Stability']),
                    fontsize=8, fontweight='bold',
                    textcoords="offset points", xytext=(8, 0))
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.4, label='Core threshold')
    ax.axvline(x=0, color='grey', linestyle=':', alpha=0.4)
    ax.set_xlabel('Mean Silhouette')
    ax.set_ylabel('Mean Bootstrap Stability')
    ax.set_title('(b) Cluster Quality Map (size = n drugs)')
    ax.legend()

    # (c) Boxplot
    ax = axes[2]
    box_data = [drug_stab[ref_labels == c] for c in clusters]
    bp = ax.boxplot(box_data, labels=[f'C{c}' for c in clusters],
                     patch_artist=True, widths=0.6)
    for idx, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.7)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Core threshold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Drug Stability Score')
    ax.set_title('(c) Stability Distribution per Cluster')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_cluster_analysis.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    # Print drugs in each cluster
    # Print drugs in each cluster
    for c in clusters:
        cd = df_drug[df_drug['Cluster'] == c]

        quality = df_stats.loc[
            df_stats['Cluster'] == c, 'Quality'
        ].iloc[0]

        print(f"\n  C{c} ({len(cd)} drugs, quality={quality}):")
        print(f"    Top 3:    {cd.head(3)[['Drug', 'Stability']].values.tolist()}")
        print(f"    Bottom 3: {cd.tail(3)[['Drug', 'Stability']].values.tolist()}")

    return df_stats, df_drug


# =============================================================
#  STEP 6: UMAP visualization
# =============================================================
def umap_plots(X_std, ref_labels, drug_names, drug_stab):
    print("\n" + "=" * 70)
    print("  STEP 6: UMAP VISUALIZATION")
    print("=" * 70)

    umap_coords = UMAP(n_neighbors=20, min_dist=0.01, n_components=2,
                        metric='cosine', random_state=42, verbose=True
                        ).fit_transform(X_std)

    n_clusters = len(set(ref_labels))
    clusters = sorted(set(ref_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # ===== Main UMAP (dual plots) =====
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Color by Cluster
    ax = axes[0]
    for idx, c in enumerate(clusters):
        m = ref_labels == c
        ax.scatter(umap_coords[m, 0], umap_coords[m, 1], c=[colors[idx]],
                   s=30, alpha=0.8, edgecolors='white', linewidth=0.3,
                   label=f'C{c} (n={np.sum(m)})')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'Leiden Clustering (k={TARGET_K})')
    sns.despine(ax=ax)

    # Color by stability
    ax = axes[1]
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=drug_stab, cmap='RdYlGn', vmin=0, vmax=1,
                     s=30, alpha=0.8, edgecolors='white', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Bootstrap Stability', shrink=0.8)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Drug Stability Score')
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_umap_main.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    # ===== Color by Core/Boundary =====
    fig, ax = plt.subplots(figsize=(10, 8))
    categories = np.array(['Core' if s >= 0.7 else 'Boundary' for s in drug_stab])
    for cat, color, alpha in [('Core', '#2E86C1', 0.8), ('Boundary', '#E74C3C', 0.6)]:
        m = categories == cat
        ax.scatter(umap_coords[m, 0], umap_coords[m, 1], c=color,
                   s=25, alpha=alpha, edgecolors='white', linewidth=0.2,
                   label=f'{cat} (n={np.sum(m)})')
    ax.legend(fontsize=10)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Core vs Boundary Drugs')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_umap_core_boundary.pdf"), dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    # ===== Interactive =====
    try:
        import plotly.express as px
        plot_df = pd.DataFrame({
            'Drug': drug_names,
            'UMAP1': umap_coords[:, 0],
            'UMAP2': umap_coords[:, 1],
            'Cluster': [f'C{c}' for c in ref_labels],
            'Stability': np.round(drug_stab,                                                                                                                                                                                                                                                                                                                                                                                                         3),
            'Category': categories
        })
        fig = px.scatter(plot_df, x='UMAP1', y='UMAP2', color='Cluster',
                          hover_name='Drug',
                          hover_data=['Stability', 'Category'],
                          opacity=0.8,
                          title=f'Leiden Clustering (k={TARGET_K}) — Interactive')
        fig.update_layout(template='simple_white')
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        fig.write_html(os.path.join(OUTPUT_DIR, "07_umap_interactive.html"))
        print("  Interactive HTML saved.")
    except ImportError:
        print("  Plotly not available, skipping interactive plot.")

    # Save coordinates
    pd.DataFrame({
        'Drug': drug_names,
        'UMAP1': umap_coords[:, 0],
        'UMAP2': umap_coords[:, 1],
        'Cluster': ref_labels,
        'Stability': np.round(drug_stab, 4),
        'Category': categories
    }).to_csv(os.path.join(OUTPUT_DIR, "07_umap_coordinates.csv"), index=False)

    return umap_coords


# =============================================================
#  STEP 7: Final report
# =============================================================
def final_report(X_std, ref_labels, drug_stab, df_stats, boot_k,
                 best_res, best_seed):
    print("\n" + "=" * 70)
    print("  STEP 7: FINAL REPORT")
    print("=" * 70)

    n_drugs = len(ref_labels)
    n_core = int(np.sum(drug_stab >= 0.7))
    n_clusters = len(set(ref_labels))
    sil = silhouette_score(X_std, ref_labels, metric='cosine')
    ch = calinski_harabasz_score(X_std, ref_labels)
    db = davies_bouldin_score(X_std, ref_labels)
    boot_mode = Counter(boot_k).most_common(1)[0]

    r = []
    r.append("=" * 70)
    r.append(f"  FINAL CLUSTERING REPORT (K={TARGET_K})")
    r.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    r.append("=" * 70)

    r.append(f"\n  Method:        Leiden (RBConfigurationVertexPartition)")
    r.append(f"  Resolution:    {best_res}")
    r.append(f"  Best Seed:     {best_seed}")
    r.append(f"  K:             {n_clusters}")
    r.append(f"  Drugs:         {n_drugs}")
    r.append(f"  Proteins:      6682")

    r.append(f"\n  INTERNAL VALIDATION:")
    r.append(f"    Silhouette:        {sil:.4f}")
    r.append(f"    Calinski-Harabasz: {ch:.2f}")
    r.append(f"    Davies-Bouldin:    {db:.4f}")

    r.append(f"\n  BOOTSTRAP STABILITY (n=300):")
    r.append(f"    Mean stability:    {np.mean(drug_stab):.4f}")
    r.append(f"    Core drugs (≥0.7): {n_core}/{n_drugs} ({n_core/n_drugs*100:.1f}%)")
    r.append(f"    Boot k mode:       {boot_mode[0]} ({boot_mode[1]}/300 = {boot_mode[1]/300*100:.1f}%)")
    r.append(f"    Boot k range:      {min(boot_k)}-{max(boot_k)}")
    r.append(f"    Boot k dist:       {dict(sorted(Counter(boot_k).items()))}")

    r.append(f"\n  PER-CLUSTER QUALITY:")
    r.append(f"    {'C':>3} {'N':>5} {'Sil':>8} {'Stab':>8} {'Core':>7} {'Quality':>8}")
    r.append(f"    {'─'*45}")
    n_high = 0
    n_medium = 0
    n_low = 0
    for _, row in df_stats.iterrows():
        r.append(f"    C{int(row['Cluster']):>2} {int(row['N_drugs']):>4} "
                 f"{row['Silhouette']:>8.3f} {row['Stability']:>8.3f} "
                 f"{row['Core_pct']:>7s} {row['Quality']:>8s}")
        if row['Quality'] == 'High':
            n_high += 1
        elif row['Quality'] == 'Medium':
            n_medium += 1
        else:
            n_low += 1

    r.append(f"\n  QUALITY SUMMARY:")
    r.append(f"    High:   {n_high}/{n_clusters}")
    r.append(f"    Medium: {n_medium}/{n_clusters}")
    r.append(f"    Low:    {n_low}/{n_clusters}")

    r.append(f"\n  OVERALL VERDICT:")
    mean_stab = np.mean(drug_stab)
    if mean_stab > 0.7 and n_high > n_clusters / 2:
        verdict = "ROBUST"
    elif mean_stab > 0.5:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"
    r.append(f"    {verdict}")

    r.append("\n" + "=" * 70)

    text = "\n".join(r)
    print(text)

    with open(os.path.join(OUTPUT_DIR, "FINAL_REPORT.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

    return text


# =============================================================
#                        Main Program
# =============================================================
if __name__ == "__main__":
    T0 = time.time()
    print("=" * 70)
    print(f"  FINAL K={TARGET_K} CLUSTERING PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---------- Read data ----------
    data = pd.read_csv(
        DATA_ROOT / "13_Final_L2FC_Matrix_Drugs_Only_367_for_UMAP.csv",
        delimiter=',', index_col=0
    )
    print(f"\nData: {data.shape[0]} drugs × {data.shape[1]} proteins")

    X_std = StandardScaler().fit_transform(data.values)
    drug_names = data.index.tolist()

    # ---------- STEP 1: Find best resolution ----------
    best_res, res_scan_df = find_best_resolution_for_k10(X_std, n_seeds=50)

    # ---------- STEP 2: Select best k=10 labels ----------
    ref_labels, best_seed = select_best_k10_labels(X_std, best_res, n_seeds=200)

    # ---------- STEP 3: Bootstrap stability ----------
    co_prob, drug_stab, boot_k = bootstrap_stability(
        X_std, best_res, ref_labels, n_bootstrap=300
    )

    # ---------- STEP 4: Multi-method consistency ----------
    methods, ari_df, nmi_df = multi_method(X_std, ref_labels)

    # ---------- STEP 5: Per-cluster/drug analysis ----------
    df_stats, df_drug = cluster_drug_analysis(X_std, ref_labels, drug_stab, drug_names)

    # ---------- STEP 6: UMAP ----------
    umap_coords = umap_plots(X_std, ref_labels, drug_names, drug_stab)

    # ---------- STEP 7: Report ----------
    final_report(X_std, ref_labels, drug_stab, df_stats, boot_k, best_res, best_seed)

    elapsed = time.time() - T0
    print(f"\n{'=' * 70}")
    print(f"  ✅ DONE! Total time: {elapsed / 60:.1f} minutes")
    print(f"  📁 Outputs: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
