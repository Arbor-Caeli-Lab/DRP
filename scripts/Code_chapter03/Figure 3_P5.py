#%%
"""
=============================================================
Cross-Cell-Line DRP Comparison — NES Fingerprint Only
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import spearmanr
import os
import time
from datetime import datetime
from pathlib import Path

np.random.seed(42)

# =============================================================
#  Path Configuration
# =============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter03"
HEPG2_GSEA_DIR = RESULTS_ROOT / "03_cluster_biology" / "enrichment"
HCT116_GSEA_DIR = RESULTS_ROOT / "04_cluster_biology_HCT116" / "enrichment"

OUTPUT_DIR = RESULTS_ROOT / "05_cross_cellline_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS_HEPG2 = 10
N_CLUSTERS_HCT116 = 9

HEPG2_DRP_NAMES = {
    1: "Cluster01",
    2: "Cluster02",
    3: "Cluster03",
    4: "Cluster04",
    5: "Cluster05",
    6: "Cluster06",
    7: "Cluster07",
    8: "Cluster08",
    9: "Cluster09",
    10: "Cluster10"
}

HCT116_DRP_NAMES = {
    1: "Cluster01",
    2: "Cluster02",
    3: "Cluster03",
    4: "Cluster04",
    5: "Cluster05",
    6: "Cluster06",
    7: "Cluster07",
    8: "Cluster08",
    9: "Cluster09"
}


def save_fig(fig_name, dpi=300):
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(OUTPUT_DIR, f"{fig_name}.{ext}"),
                    dpi=dpi, bbox_inches='tight', format=ext)
    print(f"  Saved: {fig_name}")


# =============================================================
#  Hallmark Name Standardization
# =============================================================
def normalize_hallmark_name(raw_name):
    """Standardize Hallmark pathway names to ensure term matching between the two cell lines."""
    name = str(raw_name).strip()

    # ---- Step 1: Remove all known prefixes ----
    prefixes = [
        'MSigDB_Hallmark_2020__',
        'MSigDB Hallmark 2020  ',
        'MSigDB Hallmark 2020 ',
        'msigdb_hallmark_2020__',
        'HALLMARK_',
        'hallmark_',
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
        # Also handle prefixes embedded within the string
        name = name.replace(prefix, '')

    # ---- Step 2: Standardize delimiters ----
    name = name.replace('_', ' ').strip()
    # Remove redundant spaces
    while '  ' in name:
        name = name.replace('  ', ' ')

    # ---- Step 3: Convert to lowercase and standardize known variants ----
    name_lower = name.lower().strip()

    # Correct known spelling issues
    name_lower = name_lower.replace('pperoxisome', 'peroxisome')

    # Standardize common naming variants
    replacements = {
        'tnf-alpha signaling via nf-kb': 'tnf-alpha signaling via nf-kb',
        'tnf alpha signaling via nf kb': 'tnf-alpha signaling via nf-kb',
        'tnf-alpha signaling via nf-k': 'tnf-alpha signaling via nf-kb',
        'pi3k/akt/mtor  signaling': 'pi3k-akt-mtor signaling',
        'pi3k/akt/mtor signaling': 'pi3k-akt-mtor signaling',
        'il-6/jak/stat3 signaling': 'il6-jak-stat3 signaling',
        'il-2/stat5 signaling': 'il2-stat5 signaling',
        'g2-m checkpoint': 'g2m checkpoint',
        'g2 m checkpoint': 'g2m checkpoint',
        'wnt-beta catenin signaling': 'wnt-beta-catenin signaling',
        'wnt beta catenin signaling': 'wnt-beta-catenin signaling',
        'tgf-beta signaling': 'tgf-beta signaling',
        'tgf beta signaling': 'tgf-beta signaling',
        'epithelial mesenchymal transition': 'emt',
        'epithelial mesenchymal trans': 'emt',
        'uv response up': 'uv response up',
        'uv response dn': 'uv response dn',
        'reactive oxygen species path': 'reactive oxygen species pathway',
        'reactive oxygen species pathway': 'reactive oxygen species pathway',
        'heme metabolism': 'heme metabolism',
        'kras signaling up': 'kras signaling up',
        'kras signaling dn': 'kras signaling dn',
        'myc targets v1': 'myc targets v1',
        'myc targets v2': 'myc targets v2',
    }

    for old, new in replacements.items():
        if name_lower == old:
            name_lower = new
            break

    # ---- Step 4: Convert to the canonical identifier ----
    canonical = name_lower.strip()

    return canonical


# =============================================================
#  PART 1: Build Hallmark NES Matrices
# =============================================================
def build_hallmark_nes_matrix(gsea_dir, n_clusters, cell_line_name):
    print(f"\n{'=' * 60}")
    print(f"  Building Hallmark NES matrix: {cell_line_name}")
    print(f"{'=' * 60}")

    all_nes = {}
    all_fdr = {}
    found_clusters = []

    for c in range(1, n_clusters + 1):
        fp = os.path.join(gsea_dir, f"C{c}_GSEA_prerank_FULL.csv")
        if not os.path.exists(fp):
            # Try alternative possible file names
            fp_alt = os.path.join(gsea_dir, f"C{c}_Hallmark_GSEA.csv")
            if os.path.exists(fp_alt):
                fp = fp_alt
            else:
                print(f"  Warning: No GSEA file found for C{c}")
                continue

        df = pd.read_csv(fp)
        found_clusters.append(c)

        # Automatically detect column names
        term_col = None
        nes_col = None
        fdr_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl == 'term' or cl == 'name' or cl == 'pathway':
                term_col = col
            elif cl == 'nes':
                nes_col = col
            elif 'fdr' in cl or 'q-val' in cl or 'q_val' in cl:
                fdr_col = col

        if term_col is None:
            term_col = df.columns[0]
        if nes_col is None:
            for col in df.columns:
                if 'nes' in col.lower():
                    nes_col = col
                    break
        if fdr_col is None:
            for col in df.columns:
                if 'fdr' in col.lower() or 'q-val' in col.lower():
                    fdr_col = col
                    break

        if nes_col is None:
            print(f"  Warning: Cannot find NES column in C{c}. Columns: {df.columns.tolist()}")
            continue

        print(f"  C{c}: cols=({term_col}, {nes_col}, {fdr_col}), rows={len(df)}")

        # Filter Hallmark terms
        mask = df[term_col].str.contains('Hallmark|HALLMARK|hallmark', case=False, na=False)
        df_hall = df[mask].copy()

        if len(df_hall) == 0:
            print(f"    No Hallmark terms found! Sample terms: {df[term_col].head(3).tolist()}")
            continue

        print(f"    Hallmark terms: {len(df_hall)}")

        for _, row in df_hall.iterrows():
            term_clean = normalize_hallmark_name(str(row[term_col]))
            all_nes.setdefault(term_clean, {})[c] = float(row[nes_col])
            if fdr_col:
                all_fdr.setdefault(term_clean, {})[c] = float(row[fdr_col])
            else:
                all_fdr.setdefault(term_clean, {})[c] = np.nan

    if not all_nes:
        print(f"  ERROR: No Hallmark data extracted!")
        return None, None

    # Build matrices
    nes_df = pd.DataFrame(all_nes).T
    fdr_df = pd.DataFrame(all_fdr).T

    for c in range(1, n_clusters + 1):
        if c not in nes_df.columns:
            nes_df[c] = 0.0
            fdr_df[c] = 1.0

    nes_df = nes_df[sorted(nes_df.columns)].fillna(0)
    fdr_df = fdr_df[sorted(fdr_df.columns)].fillna(1)

    nes_df.columns = [f"C{c}" for c in nes_df.columns]
    fdr_df.columns = [f"C{c}" for c in fdr_df.columns]

    print(f"\n  Final matrix: {nes_df.shape[0]} terms x {nes_df.shape[1]} clusters")
    print(f"  All terms ({len(nes_df.index)}):")
    for t in sorted(nes_df.index):
        print(f"    '{t}'")

    # Save
    nes_df.to_csv(os.path.join(OUTPUT_DIR, f"00_{cell_line_name}_NES_matrix.csv"))

    return nes_df, fdr_df


# =============================================================
#  PART 2: NES Fingerprint Cosine Similarity
# =============================================================
def compute_nes_cosine_similarity(nes_hepg2, nes_hct116):
    print(f"\n{'=' * 60}")
    print(f"  Computing NES Fingerprint Cosine Similarity")
    print(f"{'=' * 60}")

    # Identify common terms
    terms_h = set(nes_hepg2.index)
    terms_c = set(nes_hct116.index)
    common_terms = sorted(terms_h & terms_c)

    print(f"  HepG2 terms:   {len(terms_h)}")
    print(f"  HCT-116 terms: {len(terms_c)}")
    print(f"  Common terms:  {len(common_terms)}")

    only_h = terms_h - terms_c
    only_c = terms_c - terms_h
    if only_h:
        print(f"\n  Only in HepG2 ({len(only_h)}):")
        for t in sorted(only_h):
            print(f"    '{t}'")
    if only_c:
        print(f"\n  Only in HCT-116 ({len(only_c)}):")
        for t in sorted(only_c):
            print(f"    '{t}'")

    if len(common_terms) < 10:
        print("\n  *** CRITICAL: Too few common terms! ***")
        print("  Attempting fuzzy matching...")
        common_terms, nes_hepg2, nes_hct116 = fuzzy_match_terms(nes_hepg2, nes_hct116)
        if len(common_terms) < 10:
            print("  Fuzzy matching also failed. Aborting.")
            return None

    print(f"\n  Using {len(common_terms)} common Hallmark terms for comparison")

    M_hepg2 = nes_hepg2.loc[common_terms]
    M_hct116 = nes_hct116.loc[common_terms]

    # Compute cosine similarity
    n_h = M_hepg2.shape[1]
    n_c = M_hct116.shape[1]
    sim_matrix = np.zeros((n_h, n_c))

    for i, col_h in enumerate(M_hepg2.columns):
        for j, col_c in enumerate(M_hct116.columns):
            vec_h = M_hepg2[col_h].values.astype(float)
            vec_c = M_hct116[col_c].values.astype(float)
            # Handle zero vectors
            if np.allclose(vec_h, 0) or np.allclose(vec_c, 0):
                sim_matrix[i, j] = 0.0
            else:
                sim_matrix[i, j] = 1 - cosine(vec_h, vec_c)

    sim_df = pd.DataFrame(
        sim_matrix,
        index=[f"HepG2_{c}" for c in M_hepg2.columns],
        columns=[f"HCT116_{c}" for c in M_hct116.columns]
    )

    # Print results
    print(f"\n  Cosine Similarity Matrix ({n_h} x {n_c}):")
    print(sim_df.round(3).to_string())

    # Print best matches
    print(f"\n  Best matches (HepG2 → HCT-116):")
    for i, row_name in enumerate(sim_df.index):
        best_j = np.argmax(sim_df.values[i, :])
        best_val = sim_df.values[i, best_j]
        stars = '★★★' if best_val > 0.8 else '★★' if best_val > 0.5 else '★' if best_val > 0.3 else '·'
        print(f"    {row_name:15s} → {sim_df.columns[best_j]:15s}  "
              f"cos={best_val:.3f} {stars}")

    print(f"\n  Best matches (HCT-116 → HepG2):")
    for j, col_name in enumerate(sim_df.columns):
        best_i = np.argmax(sim_df.values[:, j])
        best_val = sim_df.values[best_i, j]
        stars = '★★★' if best_val > 0.8 else '★★' if best_val > 0.5 else '★' if best_val > 0.3 else '·'
        print(f"    {col_name:15s} → {sim_df.index[best_i]:15s}  "
              f"cos={best_val:.3f} {stars}")

    # Save
    sim_df.to_csv(os.path.join(OUTPUT_DIR, "01_NES_cosine_similarity.csv"))
    M_hepg2.to_csv(os.path.join(OUTPUT_DIR, "01a_HepG2_NES_common_terms.csv"))
    M_hct116.to_csv(os.path.join(OUTPUT_DIR, "01b_HCT116_NES_common_terms.csv"))

    return sim_df, M_hepg2, M_hct116, common_terms


def fuzzy_match_terms(nes_h, nes_c):
    """Attempt fuzzy term matching when exact matching yields too few common terms."""
    from difflib import SequenceMatcher

    terms_h = list(nes_h.index)
    terms_c = list(nes_c.index)
    matches = {}

    for th in terms_h:
        best_score = 0
        best_tc = None
        for tc in terms_c:
            score = SequenceMatcher(None, th, tc).ratio()
            if score > best_score:
                best_score = score
                best_tc = tc
        if best_score > 0.7:
            matches[th] = (best_tc, best_score)
            print(f"    Fuzzy: '{th}' ↔ '{best_tc}' (score={best_score:.2f})")

    if len(matches) < 10:
        return [], nes_h, nes_c

    # Rename HCT-116 terms to match HepG2 terms
    rename_map = {v[0]: k for k, v in matches.items() if v[0] in nes_c.index}
    nes_c_renamed = nes_c.rename(index=rename_map)

    common = sorted(set(nes_h.index) & set(nes_c_renamed.index))
    print(f"    Fuzzy matching recovered {len(common)} common terms")
    return common, nes_h, nes_c_renamed


# =============================================================
#  Visualization Functions
# =============================================================
def plot_cosine_similarity_heatmap(sim_df, hepg2_names, hct116_names):
    print(f"\n  Plotting cosine similarity heatmap...")

    row_labels = []
    for idx in sim_df.index:
        c_num = int(idx.split('_C')[1])
        name = hepg2_names.get(c_num, '')
        row_labels.append(f"{idx}: {name[:40]}")

    col_labels = []
    for idx in sim_df.columns:
        c_num = int(idx.split('_C')[1])
        name = hct116_names.get(c_num, '')
        col_labels.append(f"{idx}: {name[:40]}")

    fig, ax = plt.subplots(figsize=(16, 13))

    cmap = LinearSegmentedColormap.from_list(
        'sim', ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'])

    # Annotation matrix + best-match markers
    annot_str = np.empty_like(sim_df.values, dtype=object)
    for i in range(sim_df.shape[0]):
        for j in range(sim_df.shape[1]):
            annot_str[i, j] = f'{sim_df.values[i, j]:.2f}'

    # Mark row-wise best match (★) and column-wise best match (◆)
    for i in range(sim_df.shape[0]):
        best_j = np.argmax(sim_df.values[i, :])
        annot_str[i, best_j] = f'{sim_df.values[i, best_j]:.2f}★'
    for j in range(sim_df.shape[1]):
        best_i = np.argmax(sim_df.values[:, j])
        if '★' not in annot_str[best_i, j]:
            annot_str[best_i, j] = f'{sim_df.values[best_i, j]:.2f}◆'
        elif '★' in annot_str[best_i, j] and '◆' not in annot_str[best_i, j]:
            annot_str[best_i, j] = annot_str[best_i, j].replace('★', '★◆')

    sns.heatmap(
        sim_df, cmap=cmap, center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=1.5, linecolor='white',
        annot=annot_str, fmt='', annot_kws={'fontsize': 10, 'fontweight': 'bold'},
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.6},
        xticklabels=col_labels, yticklabels=row_labels
    )
    ax.set_title(
        'Cross-Cell-Line DRP Alignment: Hallmark NES Cosine Similarity\n'
        '(★ = best match for HepG2 row, ◆ = best match for HCT-116 column)',
        fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('HCT-116 Drug Response Phenotypes', fontsize=12, labelpad=10)
    ax.set_ylabel('HepG2 Drug Response Phenotypes', fontsize=12, labelpad=10)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8, rotation=0)
    plt.tight_layout()
    save_fig("01_NES_cosine_similarity_heatmap")
    plt.close()


def plot_dual_nes_heatmap(M_hepg2, M_hct116, common_terms,
                           hepg2_names, hct116_names):
    """Plot side-by-side heatmaps of NES fingerprints for the two cell lines."""
    print(f"\n  Plotting dual NES fingerprint heatmaps...")

    cmap = LinearSegmentedColormap.from_list(
        '', ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'])

    # Sort by variance across both cell lines
    combined = pd.concat([M_hepg2, M_hct116], axis=1)
    combined['var'] = combined.var(axis=1)
    terms_sorted = combined.sort_values('var', ascending=False).index.tolist()

    h_plot = M_hepg2.loc[terms_sorted]
    c_plot = M_hct116.loc[terms_sorted]

    vm = max(np.percentile(np.abs(h_plot.values), 95),
             np.percentile(np.abs(c_plot.values), 95))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, max(10, len(terms_sorted) * 0.28)),
                                    gridspec_kw={'width_ratios': [10, 9], 'wspace': 0.05})

    # HepG2
    h_labels = [f"C{int(c.replace('C', ''))}" for c in h_plot.columns]
    sns.heatmap(h_plot, cmap=cmap, center=0, vmin=-vm, vmax=vm,
                ax=ax1, linewidths=0.5,
                annot=True, fmt='.1f', annot_kws={'fontsize': 6.5},
                cbar_kws={'shrink': 0.4, 'label': 'NES'},
                xticklabels=h_labels, yticklabels=True)
    ax1.set_title('HepG2 (10 DRPs)', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=8)
    ax1.tick_params(axis='x', labelsize=9, rotation=0)
    ax1.set_ylabel('')

    # HCT-116
    c_labels = [f"C{int(c.replace('C', ''))}" for c in c_plot.columns]
    sns.heatmap(c_plot, cmap=cmap, center=0, vmin=-vm, vmax=vm,
                ax=ax2, linewidths=0.5,
                annot=True, fmt='.1f', annot_kws={'fontsize': 6.5},
                cbar_kws={'shrink': 0.4, 'label': 'NES'},
                xticklabels=c_labels, yticklabels=False)
    ax2.set_title('HCT-116 (9 DRPs)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=9, rotation=0)

    plt.suptitle('Hallmark NES Fingerprints: HepG2 vs HCT-116\n'
                 '(rows ordered by cross-cell-line variance)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig("02_dual_NES_fingerprint_heatmap")
    plt.close()


def plot_nes_profile_comparison(M_hepg2, M_hct116, common_terms,
                                 hepg2_names, hct116_names, top_pairs):
    """Visualize pathway-level Hallmark NES profiles for the top conserved DRP pairs."""
    print(f"\n  Plotting NES profile comparisons for top conserved pairs...")

    n_pairs = min(len(top_pairs), 6)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 5.5 * n_pairs))
    if n_pairs == 1:
        axes = [axes]

    for idx, (h_col, c_col, sim_val) in enumerate(top_pairs[:n_pairs]):
        ax = axes[idx]

        h_vals = M_hepg2.loc[common_terms, h_col].values
        c_vals = M_hct116.loc[common_terms, c_col].values

        # Sort by absolute HepG2 NES
        order = np.argsort(-np.abs(h_vals))
        terms_sorted = [common_terms[i] for i in order]
        h_sorted = h_vals[order]
        c_sorted = c_vals[order]

        x = np.arange(len(terms_sorted))
        width = 0.35

        h_num = int(h_col.replace('C', ''))
        c_num = int(c_col.replace('C', ''))

        ax.bar(x - width / 2, h_sorted, width,
               label=f'HepG2 {h_col}: {hepg2_names.get(h_num, "")[:35]}',
               color='#E74C3C', alpha=0.75, edgecolor='darkred', linewidth=0.5)
        ax.bar(x + width / 2, c_sorted, width,
               label=f'HCT-116 {c_col}: {hct116_names.get(c_num, "")[:35]}',
               color='#3498DB', alpha=0.75, edgecolor='darkblue', linewidth=0.5)

        # Mark pathways with discordant directions
        for k in range(len(terms_sorted)):
            if h_sorted[k] * c_sorted[k] < 0 and (abs(h_sorted[k]) > 1.3 or abs(c_sorted[k]) > 1.3):
                ax.annotate('✗', (x[k], max(abs(h_sorted[k]), abs(c_sorted[k])) + 0.15),
                            ha='center', fontsize=8, color='red', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([t[:28] for t in terms_sorted],
                           rotation=90, fontsize=6, ha='center')
        ax.set_ylabel('NES', fontsize=10)
        ax.set_title(f'HepG2-{h_col} ↔ HCT116-{c_col}  (cosine = {sim_val:.3f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Spearman correlation
        rho, _ = spearmanr(h_sorted, c_sorted)
        ax.text(0.02, 0.95, f'Spearman ρ = {rho:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_fig("03_NES_profile_comparison_top_pairs")
    plt.close()


def plot_conservation_summary(sim_df, hepg2_names, hct116_names):
    """Visualize a summary of cross-cell-line DRP conservation."""
    print(f"\n  Plotting conservation summary...")

    records = []
    for i, row_name in enumerate(sim_df.index):
        h_num = int(row_name.split('_C')[1])
        best_j = np.argmax(sim_df.values[i, :])
        c_num = int(sim_df.columns[best_j].split('_C')[1])
        sim_val = sim_df.values[i, best_j]

        # Also record the second-best match
        vals = sim_df.values[i, :].copy()
        vals[best_j] = -999
        second_j = np.argmax(vals)
        second_num = int(sim_df.columns[second_j].split('_C')[1])
        second_val = sim_df.values[i, second_j]

        records.append({
            'HepG2_DRP': f'C{h_num}',
            'HepG2_Name': hepg2_names.get(h_num, ''),
            'Best_HCT116': f'C{c_num}',
            'Best_HCT116_Name': hct116_names.get(c_num, ''),
            'Cosine': sim_val,
            'Second_HCT116': f'C{second_num}',
            'Second_Cosine': second_val,
            'Level': ('Highly conserved' if sim_val > 0.8 else
                      'Conserved' if sim_val > 0.5 else
                      'Partially conserved' if sim_val > 0.3 else
                      'Cell-line specific')
        })

    df_align = pd.DataFrame(records)
    df_align.to_csv(os.path.join(OUTPUT_DIR, "04_DRP_alignment_table.csv"), index=False)

    # Visualize as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    colors_map = {
        'Highly conserved': '#2ECC71',
        'Conserved': '#3498DB',
        'Partially conserved': '#F39C12',
        'Cell-line specific': '#E74C3C'
    }

    y_pos = range(len(df_align))
    bars = ax.barh(y_pos, df_align['Cosine'],
                    color=[colors_map[l] for l in df_align['Level']],
                    edgecolor='white', linewidth=1, height=0.7)

    # Labels
    y_labels = []
    for _, row in df_align.iterrows():
        y_labels.append(f"{row['HepG2_DRP']}: {row['HepG2_Name'][:30]}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Cosine Similarity with Best HCT-116 Match', fontsize=11)
    ax.set_title('DRP Conservation Level: HepG2 → HCT-116',
                 fontsize=13, fontweight='bold')

    # Annotate each bar with its matched DRP
    for i, (_, row) in enumerate(df_align.iterrows()):
        ax.text(row['Cosine'] + 0.01, i,
                f"→ {row['Best_HCT116']}: {row['Best_HCT116_Name'][:25]} ({row['Cosine']:.2f})",
                va='center', fontsize=7.5)

    # Threshold lines
    for thresh, label in [(0.8, 'Highly conserved'), (0.5, 'Conserved'), (0.3, 'Partial')]:
        ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.text(thresh, len(df_align) - 0.3, label, fontsize=7, ha='center',
                color='gray', style='italic')

    ax.set_xlim(-0.1, 1.15)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    save_fig("04_conservation_summary")
    plt.close()

    # Print summary report
    print(f"\n  {'=' * 70}")
    print(f"  CROSS-CELL-LINE DRP ALIGNMENT REPORT")
    print(f"  {'=' * 70}")
    for _, row in df_align.iterrows():
        print(f"  {row['HepG2_DRP']:4s} ({row['HepG2_Name'][:35]:35s})")
        print(f"    → Best:   {row['Best_HCT116']:4s} ({row['Best_HCT116_Name'][:35]}) "
              f"cos={row['Cosine']:.3f} [{row['Level']}]")
        print(f"    → Second: {row['Second_HCT116']:4s} cos={row['Second_Cosine']:.3f}")
        print()

    return df_align


def plot_bidirectional_alignment(sim_df, hepg2_names, hct116_names):
    """Plot a Sankey-style bidirectional alignment diagram using connecting lines."""
    print(f"\n  Plotting bidirectional alignment diagram...")

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, max(sim_df.shape[0], sim_df.shape[1]) + 0.5)
    ax.axis('off')

    n_h = sim_df.shape[0]
    n_c = sim_df.shape[1]

    # Node positions
    y_h = np.linspace(0, max(n_h, n_c) - 1, n_h)
    y_c = np.linspace(0, max(n_h, n_c) - 1, n_c)
    x_h = 1.5
    x_c = 8.5

    # Draw nodes
    for i, (idx, y) in enumerate(zip(sim_df.index, y_h)):
        c_num = int(idx.split('_C')[1])
        name = hepg2_names.get(c_num, '')
        ax.text(x_h, y, f"C{c_num}: {name[:30]}",
                ha='right', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8',
                          edgecolor='#E74C3C', alpha=0.8))

    for j, (idx, y) in enumerate(zip(sim_df.columns, y_c)):
        c_num = int(idx.split('_C')[1])
        name = hct116_names.get(c_num, '')
        ax.text(x_c, y, f"C{c_num}: {name[:30]}",
                ha='left', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#D4E6F1',
                          edgecolor='#3498DB', alpha=0.8))

    # Draw connections (only for cos > 0.25)
    for i in range(n_h):
        for j in range(n_c):
            sim_val = sim_df.values[i, j]
            if sim_val > 0.25:
                lw = sim_val * 4
                alpha = min(sim_val * 1.2, 0.9)
                color = '#2ECC71' if sim_val > 0.7 else '#3498DB' if sim_val > 0.4 else '#95A5A6'

                is_best_row = (j == np.argmax(sim_df.values[i, :]))
                is_best_col = (i == np.argmax(sim_df.values[:, j]))

                if is_best_row or is_best_col:
                    style = '-'
                    lw *= 1.3
                else:
                    style = '--'
                    alpha *= 0.5

                ax.plot([x_h + 0.1, x_c - 0.1], [y_h[i], y_c[j]],
                        linestyle=style, linewidth=lw, color=color, alpha=alpha)

                # Annotate similarity at the midpoint of the connection
                if is_best_row or is_best_col:
                    mid_x = (x_h + x_c) / 2
                    mid_y = (y_h[i] + y_c[j]) / 2
                    ax.text(mid_x, mid_y, f'{sim_val:.2f}',
                            ha='center', va='center', fontsize=7,
                            bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='gray', alpha=0.7))

    ax.text(x_h - 0.5, max(n_h, n_c) + 0.3, 'HepG2 DRPs',
            fontsize=14, fontweight='bold', ha='center', color='#E74C3C')
    ax.text(x_c + 0.5, max(n_h, n_c) + 0.3, 'HCT-116 DRPs',
            fontsize=14, fontweight='bold', ha='center', color='#3498DB')

    ax.set_title('Cross-Cell-Line DRP Alignment\n'
                 '(solid = best match, dashed = secondary, width ∝ similarity)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_fig("05_bidirectional_alignment")
    plt.close()


# =============================================================
#                         MAIN
# =============================================================
if __name__ == "__main__":
    T0 = time.time()
    print("=" * 70)
    print("  CROSS-CELL-LINE DRP COMPARISON — NES FINGERPRINT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ========== STEP 1: Build NES matrices ==========
    nes_hepg2, fdr_hepg2 = build_hallmark_nes_matrix(
        HEPG2_GSEA_DIR, N_CLUSTERS_HEPG2, "HepG2")
    nes_hct116, fdr_hct116 = build_hallmark_nes_matrix(
        HCT116_GSEA_DIR, N_CLUSTERS_HCT116, "HCT116")

    if nes_hepg2 is None or nes_hct116 is None:
        print("\n  ERROR: Failed to build NES matrices. Check GSEA files.")
        exit(1)

    # ========== STEP 2: Cosine similarity ==========
    result = compute_nes_cosine_similarity(nes_hepg2, nes_hct116)

    if result is None:
        print("\n  ERROR: Cosine similarity computation failed.")
        exit(1)

    sim_df, M_hepg2, M_hct116, common_terms = result

    # ========== STEP 3: Visualization ==========

    # 3a: Cosine similarity heatmap
    plot_cosine_similarity_heatmap(sim_df, HEPG2_DRP_NAMES, HCT116_DRP_NAMES)

    # 3b: Side-by-side NES fingerprint heatmaps
    plot_dual_nes_heatmap(M_hepg2, M_hct116, common_terms,
                           HEPG2_DRP_NAMES, HCT116_DRP_NAMES)

    # 3c: Pathway-level NES comparison for top conserved pairs
    top_pairs = []
    for i, row_name in enumerate(sim_df.index):
        h_col = row_name.replace('HepG2_', '')
        best_j = np.argmax(sim_df.values[i, :])
        c_col = sim_df.columns[best_j].replace('HCT116_', '')
        top_pairs.append((h_col, c_col, sim_df.values[i, best_j]))
    top_pairs.sort(key=lambda x: -x[2])

    plot_nes_profile_comparison(M_hepg2, M_hct116, common_terms,
                                 HEPG2_DRP_NAMES, HCT116_DRP_NAMES, top_pairs)

    # 3d: Conservation summary
    df_align = plot_conservation_summary(sim_df, HEPG2_DRP_NAMES, HCT116_DRP_NAMES)

    # 3e: Bidirectional alignment diagram
    plot_bidirectional_alignment(sim_df, HEPG2_DRP_NAMES, HCT116_DRP_NAMES)

    # ========== Complete ==========
    elapsed = time.time() - T0
    print(f"\n{'=' * 70}")
    print(f"  DONE! Total time: {elapsed:.1f} seconds")
    print(f"  Output files in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
