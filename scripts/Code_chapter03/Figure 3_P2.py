"""
=============================================================
Structure Validation: Local Density & DBSCAN Analysis (PDF Output)
(With Consistent Cluster Colors)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
import os
from pathlib import Path

# Set Matplotlib backend and font type (Type 42 = TrueType, editable in PDF)
matplotlib.use('Agg')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ================= Configuration =================
# ★★★ Please make sure the file paths are correct ★★★
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter03"
UMAP_FILE = RESULTS_ROOT / "01_HepG2_k10_clustering" / "07_umap_coordinates.csv"
LABEL_FILE = RESULTS_ROOT / "01_HepG2_k10_clustering" / "01_final_k10_labels.csv"

OUTPUT_DIR = RESULTS_ROOT / "02_HepG2_density_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("white")


CLUSTER_COLORS = {
    1: '#8FCB6A',
    2: '#E89ABE',
    3: '#388E3C',
    4: '#E6847C',
    5: '#D32F2F',
    6: '#F5A623',
    7: '#B385C9',
    8: '#D2A679',
    9: '#6CB0C4',
    10: '#800080'
}



# ================= 1. Data Loading =================
def load_data_strict():
    print("Loading existing UMAP coordinates...")
    df_umap = pd.read_csv(UMAP_FILE)
    # Automatically identify the Drug column
    drug_col = [c for c in df_umap.columns if 'Drug' in c or 'name' in c.lower()]
    if drug_col:
        df_umap.set_index(drug_col[0], inplace=True)
    else:
        df_umap.set_index(df_umap.columns[0], inplace=True)

    # Automatically identify UMAP columns
    u1 = [c for c in df_umap.columns if '1' in c][0]
    u2 = [c for c in df_umap.columns if '2' in c][0]
    df_umap = df_umap[[u1, u2]].rename(columns={u1: 'UMAP1', u2: 'UMAP2'})

    df_lab = pd.read_csv(LABEL_FILE)
    drug_col_lab = [c for c in df_lab.columns if 'Drug' in c or 'name' in c.lower()]
    if drug_col_lab:
        df_lab.set_index(drug_col_lab[0], inplace=True)
    else:
        df_lab.set_index(df_lab.columns[0], inplace=True)

    df_final = df_umap.join(df_lab[['Cluster']], how='inner')
    print(f"Matched {len(df_final)} drugs.")
    return df_final


# ================= 2. Local Density Calculation =================
def analyze_density(df):
    print("Calculating local density...")
    coords = df[['UMAP1', 'UMAP2']].values
    nbrs = NearestNeighbors(n_neighbors=6).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    df['Local_Density'] = 1 / (dists[:, 5] + 1e-6)

    stats = df.groupby('Cluster')['Local_Density'].median().sort_values()
    stats.to_csv(os.path.join(OUTPUT_DIR, "cluster_median_density.csv"))
    return df


# ================= 3. Plot: Density Contours (Consistent Colors) =================
# ================= 3. Plot: Density Contours (Gray Background + Colorbar + Sample Count Labels) =================
def plot_density_contours(df):
    print("Plotting density contours with sample counts (PDF)...")
    fig, ax = plt.subplots(figsize=(11.5, 8))  # Slightly wider for the legend and colorbar

    # ---------- 1. Gray density contours ----------
    x, y = df['UMAP1'], df['UMAP2']
    deltaX, deltaY = (max(x) - min(x)) / 10, (max(y) - min(y)) / 10
    xx, yy = np.mgrid[min(x) - deltaX:max(x) + deltaX:100j,
    min(y) - deltaY:max(y) + deltaY:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cf = ax.contourf(xx, yy, f, cmap='Greys', alpha=0.45, levels=20)

    # ---------- 2. Calculate the sample count for each Cluster ----------
    cluster_sizes = df['Cluster'].value_counts()

    # ---------- 3. Scatter plot + center labels (show both Cluster ID + sample count) ----------
    handles = []
    for c in sorted(df['Cluster'].unique()):
        mask = df['Cluster'] == c
        color = CLUSTER_COLORS.get(c, 'grey')

        # Plot scatter points
        sc = ax.scatter(df.loc[mask, 'UMAP1'], df.loc[mask, 'UMAP2'],
                        c=[color], s=20, alpha=0.85, label=f'C{c}')
        handles.append(sc)

        # Calculate center position
        cx = df.loc[mask, 'UMAP1'].mean()
        cy = df.loc[mask, 'UMAP2'].mean()

        # [Key] Display both Cluster ID and sample count
        n = cluster_sizes[c]
        ax.text(cx, cy, f"C{c}\n(n={n})",
                fontsize=9.5,
                fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))

    # ---------- 4. Cluster legend (optional: also show sample count) ----------
    legend_labels = [f'C{c} (n={cluster_sizes[c]})' for c in sorted(df['Cluster'].unique())]
    ax.legend(handles=handles, labels=legend_labels,
              title="Cluster (sample count)",
              title_fontsize=11,
              fontsize=9.5,
              loc='upper left',
              bbox_to_anchor=(1.02, 1))

    # ---------- 5. Gray density colorbar ----------
    cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label('Kernel Density Estimate\n(Higher = denser)',
                   fontsize=11, labelpad=15, rotation=270, va='bottom')
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Cluster Distribution on Density Landscape\n(with sample counts)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "Fig_Density_Contours_Overlay_with_n.pdf")
    plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved density plot with sample counts: {save_path}")

# ================= 4. DBSCAN Comparison =================
def run_dbscan(df):
    print("Running DBSCAN check...")
    coords = df[['UMAP1', 'UMAP2']].values
    db = DBSCAN(eps=0.35, min_samples=5).fit(coords)
    df['DBSCAN_Label'] = db.labels_

    noise_frac = df.groupby('Cluster').apply(lambda x: (x['DBSCAN_Label'] == -1).mean())
    noise_frac.to_csv(os.path.join(OUTPUT_DIR, "cluster_noise_fraction.csv"))

    # Plot: noise point distribution (red crosses)
    fig, ax = plt.subplots(figsize=(10, 8))
    normal = df[df['DBSCAN_Label'] != -1]
    ax.scatter(normal['UMAP1'], normal['UMAP2'], c='lightgrey', s=10, alpha=0.5)
    noise = df[df['DBSCAN_Label'] == -1]
    ax.scatter(noise['UMAP1'], noise['UMAP2'], c='red', marker='x', s=30, label='Sparse/Transition (Noise)')

    for c in sorted(df['Cluster'].unique()):
        mask = df['Cluster'] == c
        cx, cy = df.loc[mask, 'UMAP1'].mean(), df.loc[mask, 'UMAP2'].mean()
        ax.text(cx, cy, f'C{c}', fontsize=9, color='black', alpha=0.7)

    ax.set_title("DBSCAN Noise Detection (eps=0.35)")
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_DBSCAN_Noise_Distribution.pdf"), dpi=600, format='pdf')
    plt.close()
    return noise_frac


# ================= 5. Combined Validation Figure (Consistent Colors) =================
def plot_validation_figure(df, noise_frac_series):
    print("Plotting combined validation figure (PDF)...")
    sns.set_style("whitegrid")

    density_stats = df.groupby('Cluster')['Local_Density'].median().reset_index()
    density_stats.columns = ['Cluster', 'Density']
    df_den = density_stats.sort_values('Density')

    noise_stats = noise_frac_series.reset_index()
    noise_stats.columns = ['Cluster', 'Noise_Pct']
    noise_stats['Noise_Pct'] = noise_stats['Noise_Pct'] * 100
    df_noise = noise_stats.set_index('Cluster').loc[df_den['Cluster']].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: Local density ---
    # Use CLUSTER_COLORS to color each Cluster
    bar_colors = [CLUSTER_COLORS.get(c, 'grey') for c in df_den['Cluster']]

    sns.barplot(data=df_den, x='Cluster', y='Density', palette=bar_colors, ax=axes[0], order=df_den['Cluster'])
    axes[0].set_title('A. Median Local Density (Compactness)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Density Score', fontsize=12)

    # Annotations
    max_den = df_den['Density'].max()
    axes[0].text(0, df_den.iloc[0]['Density'] + max_den * 0.05, 'Sparse\n(C6)',
                 ha='center', color='red', fontweight='bold', fontsize=9)
    axes[0].text(len(df_den) - 2, max_den + max_den * 0.05, 'Hyper-compact\n(C7, C9, C8)',
                 ha='center', color='green', fontweight='bold', fontsize=9)

    # --- Panel B: DBSCAN noise fraction ---
    # Also use CLUSTER_COLORS
    noise_bar_colors = [CLUSTER_COLORS.get(c, 'grey') for c in df_noise['Cluster']]

    sns.barplot(data=df_noise, x='Cluster', y='Noise_Pct', palette=noise_bar_colors, ax=axes[1],
                order=df_noise['Cluster'])
    axes[1].set_title('B. DBSCAN Noise Fraction (eps=0.35)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Points classified as Noise (%)', fontsize=12)

    # Annotations
    try:
        idx_c6 = list(df_noise['Cluster']).index(6)
        axes[1].text(idx_c6, 1, 'C6: 0%\n(Connected)', ha='center', color='black', fontweight='bold', fontsize=9)
    except:
        pass

    try:
        idx_c5 = list(df_noise['Cluster']).index(5)
        val_c5 = df_noise.loc[df_noise['Cluster'] == 5, 'Noise_Pct'].values[0]
        axes[1].text(idx_c5, val_c5 + 1, f'C5: {val_c5:.0f}%\n(Weak)', ha='center', color='black', fontweight='bold',
                     fontsize=9)
    except:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_Density_Validation_Combined.pdf"), dpi=600, format='pdf')
    plt.close()


# ================= Main Program =================
if __name__ == "__main__":
    df = load_data_strict()
    df = analyze_density(df)
    plot_density_contours(df)
    noise_frac_series = run_dbscan(df)
    plot_validation_figure(df, noise_frac_series)
    print(f"\nDone! All PDF results saved to: {OUTPUT_DIR}")

#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = RESULTS_ROOT / "02_HepG2_density_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)



def plot_validation_heatmap(save_basename="Fig_Density_Validation_Heatmap"):
    # ---- merge to one table ----
    den = pd.DataFrame(density_data).set_index("Cluster")
    noi = pd.DataFrame(noise_data).set_index("Cluster")
    df = den.join(noi, how="outer").reindex(range(10))
    df["Density"] = df["Density"].astype(float)
    df["Noise_Pct"] = df["Noise_Pct"].fillna(0).astype(float)

    # ---- column-wise min-max normalization for coloring (0-1) ----
    df_norm = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0) + 1e-12)

    # ---- annotation with raw values ----
    annot = pd.DataFrame(index=df.index, columns=df.columns)
    annot["Density"] = df["Density"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    annot["Noise_Pct"] = df["Noise_Pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    # ---- plot ----
    sns.set_style("white")
    plt.figure(figsize=(5.5, 6.2), dpi=600)
    ax = sns.heatmap(
        df_norm[["Density", "Noise_Pct"]],
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.6, linecolor="white",
        annot=annot[["Density", "Noise_Pct"]],
        fmt="",
        cbar_kws={"label": "Column-wise normalized score (0–1)"}
    )
    ax.set_title("Local density and DBSCAN noise by Leiden cluster", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    ax.set_xticklabels(["Median local density", "DBSCAN noise (%)"], rotation=30, ha="right")

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, save_basename + ".png")
    pdf_path = os.path.join(OUTPUT_DIR, save_basename + ".pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" ", png_path)
    print(" ", pdf_path)
    return df

if __name__ == "__main__":
    df = load_data_strict()

    # ★ Export raw UMAP points + cluster
    df.to_csv(
        os.path.join(OUTPUT_DIR, "umap_points_with_cluster.csv")
    )

    df = analyze_density(df)

    plot_density_contours(df)

    noise_frac_series = run_dbscan(df)

    plot_validation_figure(df, noise_frac_series)

    print(f"\nDone! All PDF results saved to: {OUTPUT_DIR}")
