#%%

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests

import gseapy as gp

import os
import time
from datetime import datetime
from collections import Counter
from pathlib import Path

np.random.seed(42)

# =============================================================
#  Global Configuration
# =============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter03"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter03"
OUTPUT_DIR = RESULTS_ROOT / "03_cluster_biology"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "per_cluster"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "enrichment"), exist_ok=True)

DATA_FILE = (
    DATA_ROOT / "13_Final_L2FC_Matrix_Drugs_Only_367_for_UMAP.csv"
)
MAPPING_FILE = DATA_ROOT / "uniprot_to_gene_cache.csv"
LABEL_FILE = RESULTS_ROOT / "01_HepG2_k10_clustering" / "01_final_k10_labels.csv"


def save_fig(fig_name, dpi=600):
    path = os.path.join(OUTPUT_DIR, fig_name)
    plt.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', format='pdf')
    plt.savefig(path + ".png", dpi=dpi, bbox_inches='tight', format='png')
    print(f"  Saved: {fig_name}")


# =============================================================
#  STEP 0: Load Local Mapping File + Convert Column Names
# =============================================================
def load_local_mapping(cache_file):
    print("\n" + "=" * 70)
    print("  STEP 0: LOADING LOCAL UniProt -> Gene Symbol MAPPING")
    print("=" * 70)

    df = pd.read_csv(cache_file, dtype=str)
    print(f"  File: {cache_file}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {len(df)}")

    uniprot_col = None
    gene_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if any(x in cl for x in ['uniprot', 'accession', 'entry', 'from']):
            uniprot_col = col
        if any(x in cl for x in ['gene', 'symbol', 'name', 'to']):
            gene_col = col
    if uniprot_col is None:
        uniprot_col = df.columns[0]
        print(f"  Warning: auto-detect failed for UniProt col, using '{uniprot_col}'")
    if gene_col is None:
        gene_col = df.columns[1]
        print(f"  Warning: auto-detect failed for Gene col, using '{gene_col}'")

    print(f"  UniProt column: '{uniprot_col}'")
    print(f"  Gene column:    '{gene_col}'")

    df_clean = df[[uniprot_col, gene_col]].dropna()
    df_clean = df_clean[df_clean[gene_col].str.strip() != '']
    df_clean = df_clean[df_clean[uniprot_col].str.strip() != '']
    df_clean = df_clean.drop_duplicates(subset=uniprot_col, keep='first')

    mapping = dict(zip(df_clean[uniprot_col].str.strip(),
                       df_clean[gene_col].str.strip()))
    print(f"  Valid mappings: {len(mapping)}")
    print(f"  Examples: {dict(list(mapping.items())[:5])}")
    return mapping


def convert_data_columns(data_raw, mapping):
    print(f"\n  Converting column names: UniProt -> Gene Symbol")
    original_cols = data_raw.columns.tolist()

    mapped_cols, new_names, unmapped = [], [], []
    for col in original_cols:
        cc = col.strip()
        if cc in mapping:
            mapped_cols.append(col)
            new_names.append(mapping[cc].upper())
        else:
            unmapped.append(cc)

    n_mapped = len(mapped_cols)
    n_total = len(original_cols)
    print(f"  Mapped:   {n_mapped}/{n_total} ({n_mapped / n_total * 100:.1f}%)")
    print(f"  Unmapped: {len(unmapped)}")
    if unmapped:
        print(f"    First 10 unmapped: {unmapped[:10]}")

    data_mapped = data_raw[mapped_cols].copy()
    data_mapped.columns = new_names

    dup_genes = [g for g, cnt in Counter(new_names).items() if cnt > 1]
    if dup_genes:
        print(f"  Warning: {len(dup_genes)} duplicated gene symbols -> merging by mean")
        data_mapped = data_mapped.T.groupby(level=0).mean().T

    print(f"  Final shape: {data_mapped.shape[0]} drugs x {data_mapped.shape[1]} genes")

    known = ['EGFR', 'TP53', 'AKT1', 'MTOR', 'GAPDH', 'ACTB', 'CDK2', 'MAPK1']
    found = [g for g in known if g in data_mapped.columns]
    print(f"  Sanity check - known genes found: {found}")
    return data_mapped


# =============================================================
#  STEP 1: Extract Signature Proteins for Each Cluster
#          (Relaxed Thresholds + Fallback)
# =============================================================
def extract_signature_proteins(data, labels):
    print("\n" + "=" * 70)
    print("  STEP 1: EXTRACTING SIGNATURE PROTEINS (FIXED THRESHOLDS)")
    print("=" * 70)

    clusters = sorted(set(labels))
    n_proteins = data.shape[1]
    protein_names = data.columns.tolist()
    all_signatures = {}

    for c in clusters:
        n_in = int(np.sum(labels == c))
        print(f"\n  --- Cluster {c} (n={n_in}) ---")

        mask_in = labels == c
        X_in = data.values[mask_in]
        X_out = data.values[~mask_in]

        results = []
        for j in range(n_proteins):
            vi, vo = X_in[:, j], X_out[:, j]
            mi, mo = np.mean(vi), np.mean(vo)
            diff = mi - mo
            try:
                _, pval = mannwhitneyu(vi, vo, alternative='two-sided')
            except Exception:
                pval = 1.0
            results.append({
                'Protein': protein_names[j],
                'Mean_in_cluster': round(mi, 4),
                'Mean_rest': round(mo, 4),
                'Diff': round(diff, 4),
                'P_value': pval
            })

        df_res = pd.DataFrame(results)
        _, padj, _, _ = multipletests(df_res['P_value'].values, method='fdr_bh')
        df_res['P_adjusted'] = padj
        df_res['Direction'] = np.where(df_res['Diff'] > 0, 'Up', 'Down')

        # Three significance levels
        df_res['Sig_strict'] = (df_res['P_adjusted'] < 0.05) & (np.abs(df_res['Diff']) > 0.05)
        df_res['Sig_relaxed'] = (df_res['P_adjusted'] < 0.10) & (np.abs(df_res['Diff']) > 0.03)
        df_res['Sig_nominal'] = (df_res['P_value'] < 0.05) & (np.abs(df_res['Diff']) > 0.02)

        n_strict = df_res['Sig_strict'].sum()
        n_relaxed = df_res['Sig_relaxed'].sum()
        n_nominal = df_res['Sig_nominal'].sum()
        print(f"    Strict  (padj<0.05, |d|>0.05): {n_strict}")
        print(f"    Relaxed (padj<0.10, |d|>0.03): {n_relaxed}")
        print(f"    Nominal (p<0.05,    |d|>0.02): {n_nominal}")

        if n_strict >= 10:
            sig_col, sig_level = 'Sig_strict', 'strict'
        elif n_relaxed >= 10:
            sig_col, sig_level = 'Sig_relaxed', 'relaxed'
        elif n_nominal >= 10:
            sig_col, sig_level = 'Sig_nominal', 'nominal'
        else:
            sig_col, sig_level = None, 'top_by_effect'
            print(f"    Warning: using top 100 by |Diff| as fallback")

        if sig_col:
            df_res['Significant'] = df_res[sig_col]
        else:
            df_sorted_abs = df_res.reindex(
                df_res['Diff'].abs().sort_values(ascending=False).index)
            top_idx = df_sorted_abs.head(100).index
            df_res['Significant'] = False
            df_res.loc[top_idx, 'Significant'] = True

        df_res = df_res.sort_values('Diff', ascending=False)
        n_up = ((df_res['Significant']) & (df_res['Direction'] == 'Up')).sum()
        n_down = ((df_res['Significant']) & (df_res['Direction'] == 'Down')).sum()

        print(f"    -> Level: {sig_level} | Up: {n_up}, Down: {n_down}")
        print(f"    Top5 Up:   {df_res[df_res['Direction'] == 'Up'].head(5)['Protein'].tolist()}")
        print(f"    Top5 Down: {df_res[df_res['Direction'] == 'Down'].tail(5)['Protein'].tolist()}")

        df_res.to_csv(os.path.join(OUTPUT_DIR, "per_cluster",
                                   f"C{c}_signature_proteins.csv"), index=False)

        up_sig = df_res[(df_res['Significant']) & (df_res['Direction'] == 'Up')]['Protein'].tolist()
        down_sig = df_res[(df_res['Significant']) & (df_res['Direction'] == 'Down')]['Protein'].tolist()
        top_up_ext = df_res.sort_values('Diff', ascending=False).head(150)['Protein'].tolist()
        top_down_ext = df_res.sort_values('Diff', ascending=True).head(150)['Protein'].tolist()

        all_signatures[c] = {
            'full': df_res,
            'up': up_sig,
            'down': down_sig,
            'up_extended': top_up_ext,
            'down_extended': top_down_ext,
            'n_up': n_up,
            'n_down': n_down,
            'sig_level': sig_level
        }

    summary = pd.DataFrame([{
        'Cluster': c,
        'N_drugs': int(np.sum(labels == c)),
        'Sig_level': all_signatures[c]['sig_level'],
        'N_sig_up': all_signatures[c]['n_up'],
        'N_sig_down': all_signatures[c]['n_down'],
        'N_sig_total': all_signatures[c]['n_up'] + all_signatures[c]['n_down']
    } for c in clusters])
    print(f"\n  Signature Summary:\n{summary.to_string(index=False)}")
    summary.to_csv(os.path.join(OUTPUT_DIR, "01_signature_summary.csv"), index=False)
    return all_signatures


# =============================================================
#  STEP 2: ORA (Multiple Strategies + Fallback)
# =============================================================
def pathway_enrichment(all_signatures, background_genes=None):
    print("\n" + "=" * 70)
    print("  STEP 2: PATHWAY ENRICHMENT (ORA) — MULTI-STRATEGY")
    print("=" * 70)

    libs = [
        'KEGG_2021_Human',
        'Reactome_2022',
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023',
        'GO_Cellular_Component_2023',
        'MSigDB_Hallmark_2020',
        'WikiPathway_2023_Human'
    ]

    all_enrichment = {}

    for c, sig in all_signatures.items():
        print(f"\n  --- Cluster {c} ---")
        all_enrichment[c] = {}

        for direction in ['Up', 'Down']:
            primary = sig[direction.lower()]
            extended = sig[f'{direction.lower()}_extended']

            if len(primary) >= 5:
                gene_list = primary
                strategy = 'significant'
            elif len(extended) >= 5:
                gene_list = extended[:200]
                strategy = 'extended_top'
            else:
                print(f"    {direction}: Too few genes, skip")
                all_enrichment[c][direction] = None
                continue

            gene_list_upper = [g.upper() for g in gene_list]
            print(f"    {direction} ({strategy}): {len(gene_list_upper)} genes, "
                  f"e.g. {gene_list_upper[:5]}")

            try:
                enr = gp.enrichr(
                    gene_list=gene_list_upper,
                    gene_sets=libs,
                    organism='Human',
                    outdir=None,
                    no_plot=True,
                    cutoff=1.0  # Do not pre-filter here
                )
                df_enr = enr.results.copy()

                # Tiered filtering
                df_s = df_enr[df_enr['Adjusted P-value'] < 0.05]
                df_r = df_enr[df_enr['Adjusted P-value'] < 0.10]
                df_n = df_enr[df_enr['P-value'] < 0.05]

                if len(df_s) >= 3:
                    df_final, flevel = df_s, 'padj<0.05'
                elif len(df_r) >= 3:
                    df_final, flevel = df_r, 'padj<0.10'
                elif len(df_n) >= 3:
                    df_final, flevel = df_n, 'p<0.05(nominal)'
                else:
                    df_final = df_enr.sort_values('P-value').head(10)
                    flevel = 'top10(no sig)'

                df_final = df_final.sort_values('Adjusted P-value')
                all_enrichment[c][direction] = df_final

                df_final.to_csv(os.path.join(
                    OUTPUT_DIR, "enrichment",
                    f"C{c}_{direction}_enrichment.csv"), index=False)

                print(f"      -> {len(df_final)} terms ({flevel})")
                for _, row in df_final.head(5).iterrows():
                    print(f"        {row['Gene_set']}: {row['Term'][:50]} "
                          f"(padj={row['Adjusted P-value']:.2e})")

            except Exception as e:
                print(f"      -> Failed: {e}")
                # Fallback: retry with extended list + smaller library set
                if strategy == 'significant' and len(extended) > 20:
                    print(f"      -> Retrying with extended list...")
                    try:
                        enr2 = gp.enrichr(
                            gene_list=[g.upper() for g in extended[:150]],
                            gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human'],
                            organism='Human',
                            outdir=None, no_plot=True, cutoff=1.0
                        )
                        df2 = enr2.results.sort_values('P-value').head(15)
                        all_enrichment[c][direction] = df2
                        df2.to_csv(os.path.join(
                            OUTPUT_DIR, "enrichment",
                            f"C{c}_{direction}_enrichment_fallback.csv"), index=False)
                        print(f"      -> Fallback: {len(df2)} terms")
                    except Exception as e2:
                        print(f"      -> Fallback also failed: {e2}")
                        all_enrichment[c][direction] = None
                else:
                    all_enrichment[c][direction] = None

    return all_enrichment


# =============================================================
#  STEP 2b: GSEA Prerank (Comprehensive Display + By Library)
# =============================================================
def gsea_prerank_analysis(all_signatures):
    print("\n" + "=" * 70)
    print("  STEP 2b: GSEA PRERANK ANALYSIS (COMPREHENSIVE)")
    print("=" * 70)

    gene_sets = [
        'KEGG_2021_Human',
        'MSigDB_Hallmark_2020',
        'Reactome_2022',
        'GO_Biological_Process_2023'
    ]

    gsea_results = {}
    gsea_by_library = {}

    for c, sig in all_signatures.items():
        print(f"\n  {'=' * 55}")
        print(f"  CLUSTER {c}")
        print(f"  {'=' * 55}")

        df_full = sig['full'].copy()
        df_full['Protein'] = df_full['Protein'].str.upper()

        rnk = df_full.set_index('Protein')['Diff'].dropna()
        rnk = rnk[~rnk.index.duplicated(keep='first')]

        np.random.seed(42)
        rnk = rnk + np.random.uniform(-1e-10, 1e-10, size=len(rnk))
        rnk = rnk.sort_values(ascending=False)

        if len(rnk) < 100:
            print(f"    Too few genes ({len(rnk)}), skip")
            gsea_results[c] = None
            gsea_by_library[c] = {}
            continue

        print(f"    Ranked list: {len(rnk)} genes")

        try:
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets,
                min_size=10,
                max_size=500,
                permutation_num=1000,
                outdir=None,
                no_plot=True,
                seed=42,
                verbose=False
            )

            df_gsea = pre_res.res2d.copy()
            df_gsea.to_csv(os.path.join(
                OUTPUT_DIR, "enrichment",
                f"C{c}_GSEA_prerank_FULL.csv"), index=False)

            df_sig = df_gsea[df_gsea['FDR q-val'] < 0.25].copy()
            df_sig_strict = df_gsea[df_gsea['FDR q-val'] < 0.05].copy()

            print(f"    Total tested: {len(df_gsea)}")
            print(f"    FDR<0.25: {len(df_sig)} | FDR<0.05: {len(df_sig_strict)}")

            gsea_results[c] = df_sig
            gsea_by_library[c] = {}

            # ---------- Display by library ----------
            # Detect library identifier column
            has_geneset_col = 'Gene_set' in df_gsea.columns

            lib_filters = {
                'MSigDB_Hallmark_2020': lambda t: 'HALLMARK' in str(t).upper(),
                'KEGG_2021_Human': lambda t: (
                    'HALLMARK' not in str(t).upper()
                    and 'REACTOME' not in str(t).upper()
                    and not str(t).upper().startswith('GO_')
                ),
                'Reactome_2022': lambda t: 'REACTOME' in str(t).upper() or 'R-HSA' in str(t).upper(),
                'GO_Biological_Process_2023': lambda t: str(t).upper().startswith('GO_') or 'GO:' in str(t).upper()
            }

            for lib in gene_sets:
                if has_geneset_col:
                    df_lib = df_gsea[df_gsea['Gene_set'] == lib].copy()
                elif lib in lib_filters:
                    df_lib = df_gsea[df_gsea['Term'].apply(lib_filters[lib])].copy()
                else:
                    continue

                if len(df_lib) == 0:
                    continue

                gsea_by_library[c][lib] = df_lib
                df_lib_sig = df_lib[df_lib['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
                lib_short = lib.split('_')[0]

                if len(df_lib_sig) > 0:
                    print(f"\n    [{lib_short}] {len(df_lib_sig)} sig terms:")
                    act = df_lib_sig[df_lib_sig['NES'] > 0]
                    sup = df_lib_sig[df_lib_sig['NES'] < 0]
                    if len(act) > 0:
                        print(f"      Activated ({len(act)}):")
                        for _, r in act.head(5).iterrows():
                            print(f"        UP {r['Term'][:55]} "
                                  f"NES={r['NES']:.2f} FDR={r['FDR q-val']:.3f}")
                    if len(sup) > 0:
                        print(f"      Suppressed ({len(sup)}):")
                        for _, r in sup.tail(5).iterrows():
                            print(f"        DN {r['Term'][:55]} "
                                  f"NES={r['NES']:.2f} FDR={r['FDR q-val']:.3f}")

            # ---------- Display all Hallmark terms ----------
            if has_geneset_col:
                df_hall = df_gsea[df_gsea['Gene_set'] == 'MSigDB_Hallmark_2020'].copy()
            else:
                df_hall = df_gsea[df_gsea['Term'].str.contains(
                    'HALLMARK', case=False, na=False)].copy()

            if len(df_hall) > 0:
                df_hall = df_hall.sort_values('NES', ascending=False)
                print(f"\n    HALLMARK FULL ({len(df_hall)} terms):")
                for _, r in df_hall.iterrows():
                    fdr = r['FDR q-val']
                    stars = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.25 else ''
                    arrow = 'UP' if r['NES'] > 0 else 'DN'
                    print(f"      {arrow} {r['Term'][:50]:50s} "
                          f"NES={r['NES']:+.2f} FDR={fdr:.3f} {stars}")
                df_hall.to_csv(os.path.join(
                    OUTPUT_DIR, "enrichment",
                    f"C{c}_Hallmark_GSEA.csv"), index=False)

        except Exception as e:
            print(f"    GSEA failed: {e}")
            gsea_results[c] = None
            gsea_by_library[c] = {}

    return gsea_results, gsea_by_library


# =============================================================
#  STEP 3: Signature Protein Heatmap
# =============================================================
def plot_signature_heatmap(data, labels, all_signatures, top_n=20):
    print("\n" + "=" * 70)
    print("  STEP 3: SIGNATURE HEATMAP")
    print("=" * 70)

    clusters = sorted(set(labels))
    top_proteins = []
    for c in clusters:
        df = all_signatures[c]['full']
        sig_df = df[df['Significant']].copy()
        top_proteins.extend(
            sig_df[sig_df['Direction'] == 'Up'].head(top_n // 2)['Protein'].tolist())
        top_proteins.extend(
            sig_df[sig_df['Direction'] == 'Down'].tail(top_n // 2)['Protein'].tolist())

    seen = set()
    unique_proteins = [p for p in top_proteins if p not in seen and not seen.add(p)]
    unique_proteins = [p for p in unique_proteins if p in data.columns]

    if not unique_proteins:
        print("  Warning: no proteins for heatmap")
        return None

    if len(unique_proteins) > 150:
        reduced = []
        for c in clusters:
            df = all_signatures[c]['full']
            sd = df[df['Significant']]
            reduced.extend(sd[sd['Direction'] == 'Up'].head(5)['Protein'].tolist())
            reduced.extend(sd[sd['Direction'] == 'Down'].tail(5)['Protein'].tolist())
        seen2 = set()
        unique_proteins = [p for p in reduced if p not in seen2 and not seen2.add(p)]
        unique_proteins = [p for p in unique_proteins if p in data.columns]

    print(f"  Proteins: {len(unique_proteins)}")

    hm = pd.DataFrame(index=unique_proteins)
    for c in clusters:
        mask = labels == c
        hm[f'C{c}\n(n={np.sum(mask)})'] = data.loc[mask, unique_proteins].mean()

    fig_h = max(8, len(hm) * 0.15)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    cmap = LinearSegmentedColormap.from_list('', ['#2166AC', '#F7F7F7', '#B2182B'])
    vm = np.percentile(np.abs(hm.values), 95)

    sns.heatmap(hm, cmap=cmap, center=0, vmin=-vm, vmax=vm, ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Mean log2FC', 'shrink': 0.5},
                yticklabels=True, xticklabels=True)
    ax.set_title('Cluster Signature Proteins\n(Top up/down per cluster)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=10)
    plt.tight_layout()
    save_fig("02_signature_heatmap")
    plt.close()
    hm.to_csv(os.path.join(OUTPUT_DIR, "02_signature_heatmap_data.csv"))
    return hm


# =============================================================
#  STEP 4: Enrichment Visualization
# =============================================================
def plot_enrichment_summary(all_enrichment, gsea_results, gsea_by_library, labels):
    print("\n" + "=" * 70)
    print("  STEP 4: ENRICHMENT VISUALIZATION (IMPROVED)")
    print("=" * 70)

    clusters = sorted(set(labels))
    cmap_div = LinearSegmentedColormap.from_list('', ['#2166AC', '#F7F7F7', '#B2182B'])

    # --- A. ORA bubble per cluster ---
    for c in clusters:
        for direction in ['Up', 'Down']:
            enr = all_enrichment.get(c, {}).get(direction)
            if enr is None or len(enr) == 0:
                continue

            df_plot = enr.head(20).copy()
            if len(df_plot) == 0:
                continue

            df_plot['neg_log10_padj'] = -np.log10(
                df_plot['Adjusted P-value'].clip(1e-50))
            df_plot['Gene_Ratio'] = df_plot['Overlap'].apply(
                lambda x: int(str(x).split('/')[0]) / max(int(str(x).split('/')[1]), 1)
                if '/' in str(x) else 0)
            df_plot['N_genes'] = df_plot['Overlap'].apply(
                lambda x: int(str(x).split('/')[0]) if '/' in str(x) else 0)
            df_plot['Term_short'] = df_plot['Term'].apply(
                lambda x: (str(x)[:55] + '...') if len(str(x)) > 55 else str(x))

            if 'Gene_set' in df_plot.columns:
                df_plot['Label'] = df_plot.apply(
                    lambda r: f"[{str(r['Gene_set'])[:8]}] {r['Term_short']}", axis=1)
            else:
                df_plot['Label'] = df_plot['Term_short']

            fig, ax = plt.subplots(figsize=(12, max(5, len(df_plot) * 0.4)))
            sc = ax.scatter(
                df_plot['neg_log10_padj'], range(len(df_plot)),
                s=df_plot['N_genes'] * 20 + 30, c=df_plot['Gene_Ratio'],
                cmap='YlOrRd', edgecolors='black', linewidth=0.5, alpha=0.8,
                vmin=0, vmax=max(df_plot['Gene_Ratio'].max(), 0.1))
            ax.set_yticks(range(len(df_plot)))
            ax.set_yticklabels(df_plot['Label'], fontsize=8)
            ax.set_xlabel('-log10(adjusted P-value)', fontsize=11)
            ax.set_title(f'Cluster {c} - {direction}-regulated Pathways (ORA)',
                         fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            plt.colorbar(sc, ax=ax, label='Gene Ratio', shrink=0.7)
            plt.tight_layout()
            save_fig(f"enrichment/C{c}_{direction}_bubble")
            plt.close()

    # --- B. Combined Hallmark NES heatmap ---
    hallmark_nes, hallmark_fdr = {}, {}
    for c in clusters:
        fp = os.path.join(OUTPUT_DIR, "enrichment", f"C{c}_GSEA_prerank_FULL.csv")
        if os.path.exists(fp):
            df_full = pd.read_csv(fp)
        elif gsea_results.get(c) is not None and len(gsea_results[c]) > 0:
            df_full = gsea_results[c]
        else:
            continue

        hall = df_full[df_full['Term'].str.contains('HALLMARK', case=False, na=False)]
        for _, r in hall.iterrows():
            t = str(r['Term']).replace('HALLMARK_', '').replace('_', ' ')
            hallmark_nes.setdefault(t, {})[c] = r['NES']
            hallmark_fdr.setdefault(t, {})[c] = r['FDR q-val']

    if hallmark_nes:
        nes_df = pd.DataFrame(hallmark_nes).T
        fdr_df = pd.DataFrame(hallmark_fdr).T
        for c in clusters:
            if c not in nes_df.columns:
                nes_df[c] = 0
                fdr_df[c] = 1
        nes_df = nes_df[sorted(nes_df.columns)].fillna(0)
        fdr_df = fdr_df[sorted(fdr_df.columns)].fillna(1)
        nes_df.columns = [f'C{c}' for c in nes_df.columns]
        fdr_df.columns = [f'C{c}' for c in fdr_df.columns]

        nes_df['var'] = nes_df.var(axis=1)
        nes_df = nes_df.sort_values('var', ascending=False)
        top_terms = nes_df.head(50).index
        nes_plot = nes_df.loc[top_terms].drop('var', axis=1)
        fdr_plot = fdr_df.loc[fdr_df.index.isin(top_terms)]

        # Annotation matrix
        annot = nes_plot.copy().astype(str)
        for col in nes_plot.columns:
            for term in nes_plot.index:
                nv = nes_plot.loc[term, col]
                fv = fdr_plot.loc[term, col] if term in fdr_plot.index else 1
                s = '***' if fv < 0.01 else '**' if fv < 0.05 else '*' if fv < 0.25 else ''
                annot.loc[term, col] = f'{nv:.1f}{s}'

        fig, ax = plt.subplots(figsize=(14, max(8, len(nes_plot) * 0.35)))
        vm = np.percentile(np.abs(nes_plot.values), 95)
        sns.heatmap(nes_plot, cmap=cmap_div, center=0, vmin=-vm, vmax=vm,
                    ax=ax, linewidths=0.5, annot=annot.values, fmt='',
                    annot_kws={'fontsize': 7},
                    cbar_kws={'label': 'NES', 'shrink': 0.5})
        ax.set_title('MSigDB Hallmark Pathways - NES across Clusters\n'
                     '(*** FDR<0.01, ** FDR<0.05, * FDR<0.25)',
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10)
        plt.tight_layout()
        save_fig("03_gsea_hallmark_heatmap")
        plt.close()
        nes_plot.to_csv(os.path.join(OUTPUT_DIR, "03_gsea_hallmark_nes.csv"))
        fdr_plot.to_csv(os.path.join(OUTPUT_DIR, "03_gsea_hallmark_fdr.csv"))
        print(f"  Hallmark heatmap: {nes_plot.shape}")

    # --- C. KEGG NES heatmap ---
    kegg_nes = {}
    for c in clusters:
        fp = os.path.join(OUTPUT_DIR, "enrichment", f"C{c}_GSEA_prerank_FULL.csv")
        if not os.path.exists(fp):
            continue
        df_full = pd.read_csv(fp)
        kegg = df_full[~df_full['Term'].str.contains(
            'HALLMARK|Reactome|GO_', case=False, na=False)]
        kegg = kegg[kegg['FDR q-val'] < 0.25]
        for _, r in kegg.iterrows():
            t = str(r['Term']).replace('_', ' ')[:45]
            kegg_nes.setdefault(t, {})[c] = r['NES']

    if len(kegg_nes) > 5:
        kdf = pd.DataFrame(kegg_nes).T.fillna(0)
        kdf.columns = [f'C{c}' for c in kdf.columns]
        kdf['var'] = kdf.var(axis=1)
        kdf = kdf.sort_values('var', ascending=False).head(30).drop('var', axis=1)

        fig, ax = plt.subplots(figsize=(14, max(6, len(kdf) * 0.3)))
        sns.heatmap(kdf, cmap=cmap_div, center=0, ax=ax, linewidths=0.5,
                    annot=True, fmt='.1f', annot_kws={'fontsize': 7},
                    cbar_kws={'label': 'NES', 'shrink': 0.5})
        ax.set_title('KEGG Pathways - NES across Clusters (FDR<0.25)',
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        save_fig("03b_gsea_kegg_heatmap")
        plt.close()

    return hallmark_nes


# =============================================================
#  STEP 5: Cluster Mean Profile Heatmap
# =============================================================
def plot_cluster_profile_heatmap(data, labels):
    print("\n" + "=" * 70)
    print("  STEP 5: CLUSTER MEAN PROFILE HEATMAP")
    print("=" * 70)

    clusters = sorted(set(labels))
    cm = pd.DataFrame(index=data.columns)
    for c in clusters:
        mask = labels == c
        cm[f'C{c} (n={np.sum(mask)})'] = data.loc[mask].mean()

    cm['var'] = cm.var(axis=1)
    top_p = cm.sort_values('var', ascending=False).head(100).index
    pd_plot = cm.loc[top_p].drop('var', axis=1)

    fig, ax = plt.subplots(figsize=(12, 20))
    cmap = LinearSegmentedColormap.from_list('', ['#2166AC', '#F7F7F7', '#B2182B'])
    vm = np.percentile(np.abs(pd_plot.values), 95)
    sns.heatmap(pd_plot, cmap=cmap, center=0, vmin=-vm, vmax=vm, ax=ax,
                linewidths=0, yticklabels=True,
                cbar_kws={'label': 'Mean log2FC', 'shrink': 0.3})
    ax.set_title('Top 100 Variable Proteins across Clusters',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=5)
    ax.tick_params(axis='x', labelsize=10)
    plt.tight_layout()
    save_fig("04_cluster_mean_profile")
    plt.close()
    cm.drop('var', axis=1).to_csv(
        os.path.join(OUTPUT_DIR, "04_cluster_mean_profiles.csv"))


# =============================================================
#  STEP 6: Comprehensive Report + Hallmark-Priority Naming
# =============================================================
def generate_biology_report(all_signatures, all_enrichment, gsea_results,
                            gsea_by_library, labels, drug_names):
    print("\n" + "=" * 70)
    print("  STEP 6: BIOLOGY REPORT & HALLMARK-PRIORITY NAMING")
    print("=" * 70)

    clusters = sorted(set(labels))
    rpt = []
    rpt.append("=" * 70)
    rpt.append("  CLUSTER BIOLOGY ANNOTATION REPORT")
    rpt.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.append("=" * 70)

    naming = {}

    for c in clusters:
        n_drugs = int(np.sum(labels == c))
        rpt.append(f"\n{'=' * 60}")
        rpt.append(f"  CLUSTER {c} (n = {n_drugs} drugs)")
        rpt.append(f"{'=' * 60}")

        drugs_in = [drug_names[i] for i in range(len(labels)) if labels[i] == c]
        rpt.append(f"\n  Drugs: {', '.join(drugs_in[:10])}")
        if len(drugs_in) > 10:
            rpt.append(f"         ... and {len(drugs_in) - 10} more")

        sig = all_signatures[c]
        rpt.append(f"\n  Signature Proteins (level: {sig['sig_level']}):")
        rpt.append(f"    Up:   {sig['n_up']}")
        rpt.append(f"    Down: {sig['n_down']}")
        if sig['up']:
            rpt.append(f"    Top Up:   {', '.join(sig['up'][:10])}")
        if sig['down']:
            rpt.append(f"    Top Down: {', '.join(sig['down'][:10])}")

        # ---- ORA ----
        rpt.append(f"\n  --- ORA Enrichment ---")
        for d in ['Up', 'Down']:
            enr = all_enrichment.get(c, {}).get(d)
            if enr is not None and len(enr) > 0:
                rpt.append(f"    {d}-regulated ({len(enr)} terms):")
                for _, row in enr.head(10).iterrows():
                    pa = row['Adjusted P-value']
                    sm = '***' if pa < 0.001 else '**' if pa < 0.01 else \
                         '*' if pa < 0.05 else '~' if pa < 0.1 else ''
                    gs = str(row.get('Gene_set', ''))[:12]
                    rpt.append(f"      {row['Term'][:55]} "
                               f"(padj={pa:.2e}{sm}, {gs})")
            else:
                rpt.append(f"    {d}: No terms")

        # ---- GSEA by library ----
        rpt.append(f"\n  --- GSEA Prerank ---")

        fp = os.path.join(OUTPUT_DIR, "enrichment", f"C{c}_GSEA_prerank_FULL.csv")
        if os.path.exists(fp):
            df_gsea_full = pd.read_csv(fp)
        elif gsea_results.get(c) is not None:
            df_gsea_full = gsea_results[c]
        else:
            df_gsea_full = None

        hallmark_terms = []
        kegg_terms = []

        if df_gsea_full is not None and len(df_gsea_full) > 0:
            # Hallmark
            df_hall = df_gsea_full[df_gsea_full['Term'].str.contains(
                'HALLMARK', case=False, na=False)].sort_values('NES', ascending=False)
            if len(df_hall) > 0:
                rpt.append(f"\n    HALLMARK ({len(df_hall)} terms):")
                for _, r in df_hall.iterrows():
                    fdr = r['FDR q-val']
                    s = '***' if fdr < 0.01 else '**' if fdr < 0.05 else '*' if fdr < 0.25 else ''
                    ar = 'UP' if r['NES'] > 0 else 'DN'
                    rpt.append(f"      {ar} {r['Term'][:50]:50s} "
                               f"NES={r['NES']:+.2f} FDR={fdr:.3f} {s}")
                    if fdr < 0.25:
                        hallmark_terms.append({
                            'term': str(r['Term']).replace('HALLMARK_', ''),
                            'NES': r['NES'], 'FDR': fdr})

            # KEGG
            df_kegg = df_gsea_full[~df_gsea_full['Term'].str.contains(
                'HALLMARK|Reactome|GO_', case=False, na=False)]
            dk_sig = df_kegg[df_kegg['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
            if len(dk_sig) > 0:
                rpt.append(f"\n    KEGG ({len(dk_sig)} sig terms):")
                for _, r in dk_sig.head(10).iterrows():
                    ar = 'UP' if r['NES'] > 0 else 'DN'
                    rpt.append(f"      {ar} {r['Term'][:50]} "
                               f"(NES={r['NES']:+.2f}, FDR={r['FDR q-val']:.3f})")
                    kegg_terms.append({
                        'term': str(r['Term']), 'NES': r['NES'],
                        'FDR': r['FDR q-val']})

            # Reactome
            df_react = df_gsea_full[df_gsea_full['Term'].str.contains(
                'Reactome|R-HSA', case=False, na=False)]
            dr_sig = df_react[df_react['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
            if len(dr_sig) > 0:
                rpt.append(f"\n    Reactome ({len(dr_sig)} sig terms):")
                for _, r in dr_sig.head(10).iterrows():
                    ar = 'UP' if r['NES'] > 0 else 'DN'
                    rpt.append(f"      {ar} {r['Term'][:50]} "
                               f"(NES={r['NES']:+.2f}, FDR={r['FDR q-val']:.3f})")

            # GO
            df_go = df_gsea_full[df_gsea_full['Term'].str.contains(
                'GO_', case=False, na=False)]
            dg_sig = df_go[df_go['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
            if len(dg_sig) > 0:
                rpt.append(f"\n    GO ({len(dg_sig)} sig terms):")
                for _, r in dg_sig.head(8).iterrows():
                    ar = 'UP' if r['NES'] > 0 else 'DN'
                    rpt.append(f"      {ar} {r['Term'][:50]} "
                               f"(NES={r['NES']:+.2f}, FDR={r['FDR q-val']:.3f})")
        else:
            rpt.append(f"    No GSEA results")

        # ---- Hallmark-priority naming ----
        suggested_name = f"Cluster_{c}"
        naming_source = 'default'

        # P1: Hallmark
        if hallmark_terms:
            best = sorted(hallmark_terms, key=lambda x: -abs(x['NES']))[:2]
            parts = []
            for ht in best:
                tc = ht['term'].replace('_', ' ').title()
                dr = 'Up' if ht['NES'] > 0 else 'Dn'
                parts.append(f"{tc} {dr}")
            suggested_name = f"C{c}: {' / '.join(parts)}"
            naming_source = 'Hallmark'
        # P2: KEGG
        elif kegg_terms:
            bk = sorted(kegg_terms, key=lambda x: -abs(x['NES']))[0]
            tc = bk['term'].replace('_', ' ')[:35]
            dr = 'Up' if bk['NES'] > 0 else 'Dn'
            suggested_name = f"C{c}: {tc} {dr}"
            naming_source = 'KEGG'
        # P3: ORA
        elif any(all_enrichment.get(c, {}).get(d) is not None and
                 len(all_enrichment.get(c, {}).get(d, pd.DataFrame())) > 0
                 for d in ['Up', 'Down']):
            for d in ['Up', 'Down']:
                enr = all_enrichment.get(c, {}).get(d)
                if enr is not None and len(enr) > 0:
                    suggested_name = f"C{c}: {enr.iloc[0]['Term'][:35]} ({d})"
                    naming_source = 'ORA'
                    break
        # P4: Signature genes
        elif sig['up']:
            suggested_name = f"C{c}: {'/'.join(sig['up'][:3])} Up"
            naming_source = 'signature_genes'

        naming[c] = {
            'suggested_name': suggested_name,
            'naming_source': naming_source,
            'hallmark_terms': hallmark_terms[:5],
            'kegg_terms': kegg_terms[:5],
            'n_drugs': n_drugs,
            'n_sig_up': sig['n_up'],
            'n_sig_down': sig['n_down']
        }

        rpt.append(f"\n  >> SUGGESTED NAME: {suggested_name}")
        rpt.append(f"     (Source: {naming_source})")

    # ---- Summary ----
    rpt.append(f"\n\n{'=' * 70}")
    rpt.append("  CLUSTER NAMING SUMMARY (Hallmark Priority)")
    rpt.append(f"{'=' * 70}")
    rpt.append(f"  {'Cluster':<10} {'N':>4}  {'Source':<15}  Name")
    rpt.append(f"  {'-' * 10} {'-' * 4}  {'-' * 15}  {'-' * 40}")
    for c in clusters:
        ns = naming[c]
        rpt.append(f"  C{c:<9} {ns['n_drugs']:>4}  "
                    f"{ns['naming_source']:<15}  {ns['suggested_name']}")

    # ---- Hallmark fingerprint ----
    rpt.append(f"\n\n{'=' * 70}")
    rpt.append("  HALLMARK FINGERPRINT PER CLUSTER")
    rpt.append(f"{'=' * 70}")
    for c in clusters:
        ns = naming[c]
        if ns['hallmark_terms']:
            rpt.append(f"\n  Cluster {c}:")
            for ht in ns['hallmark_terms']:
                ar = 'UP' if ht['NES'] > 0 else 'DN'
                s = '***' if ht['FDR'] < 0.01 else '**' if ht['FDR'] < 0.05 else '*'
                tc = ht['term'].replace('_', ' ')
                rpt.append(f"    {ar} {tc:45s} NES={ht['NES']:+.2f} {s}")
        else:
            rpt.append(f"\n  Cluster {c}: No significant Hallmark terms")
    rpt.append("\n" + "=" * 70)

    text = "\n".join(rpt)
    print(text)

    with open(os.path.join(OUTPUT_DIR, "BIOLOGY_REPORT.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

    df_naming = pd.DataFrame([{
        'Cluster': c,
        'N_drugs': v['n_drugs'],
        'Naming_Source': v['naming_source'],
        'Suggested_Name': v['suggested_name'],
        'Hallmark_Terms': '; '.join(
            [f"{h['term']}(NES={h['NES']:+.2f})" for h in v['hallmark_terms'][:3]]),
        'KEGG_Terms': '; '.join(
            [f"{k['term'][:30]}(NES={k['NES']:+.2f})" for k in v['kegg_terms'][:3]]),
        'N_sig_up': v['n_sig_up'],
        'N_sig_down': v['n_sig_down']
    } for c, v in naming.items()])
    df_naming.to_csv(os.path.join(OUTPUT_DIR, "05_cluster_naming.csv"), index=False)

    # Fingerprint matrix
    all_ht = set()
    for ns in naming.values():
        for ht in ns['hallmark_terms']:
            all_ht.add(ht['term'].replace('_', ' '))
    if all_ht:
        fp = {}
        for c in clusters:
            fp[f'C{c}'] = {}
            for ht in naming[c]['hallmark_terms']:
                fp[f'C{c}'][ht['term'].replace('_', ' ')] = ht['NES']
        pd.DataFrame(fp).fillna(0).to_csv(
            os.path.join(OUTPUT_DIR, "05b_hallmark_fingerprint.csv"))

    return naming


# =============================================================
#  STEP 7: Hallmark Fingerprint Visualization
# =============================================================
def plot_hallmark_fingerprint(gsea_results, labels):
    print("\n" + "=" * 70)
    print("  STEP 7: HALLMARK FINGERPRINT VISUALIZATION")
    print("=" * 70)

    clusters = sorted(set(labels))
    all_nes, all_fdr = {}, {}

    for c in clusters:
        fp = os.path.join(OUTPUT_DIR, "enrichment", f"C{c}_GSEA_prerank_FULL.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
        else:
            continue
        hall = df[df['Term'].str.contains('HALLMARK', case=False, na=False)]
        for _, r in hall.iterrows():
            t = str(r['Term']).replace('HALLMARK_', '').replace('_', ' ')
            all_nes.setdefault(t, {})[c] = r['NES']
            all_fdr.setdefault(t, {})[c] = r['FDR q-val']

    if not all_nes:
        print("  No Hallmark data")
        return

    nes_df = pd.DataFrame(all_nes).T
    fdr_df = pd.DataFrame(all_fdr).T
    for c in clusters:
        if c not in nes_df.columns:
            nes_df[c] = 0
            fdr_df[c] = 1
    nes_df = nes_df[sorted(nes_df.columns)].fillna(0)
    fdr_df = fdr_df[sorted(fdr_df.columns)].fillna(1)
    nes_df.columns = [f'C{c}' for c in nes_df.columns]
    fdr_df.columns = [f'C{c}' for c in fdr_df.columns]

    # Fingerprint
    fingerprint = {}
    for col in nes_df.columns:
        top_t = nes_df[col].abs().idxmax()
        fingerprint[col] = {'term': top_t, 'NES': nes_df.loc[top_t, col]}

    # Hierarchical clustering on rows
    if len(nes_df) > 2:
        Z = linkage(nes_df.values, method='ward')
        order = leaves_list(Z)
        nes_df = nes_df.iloc[order]

    # Annotation
    annot_arr = np.empty_like(nes_df.values, dtype=object)
    for i, term in enumerate(nes_df.index):
        for j, col in enumerate(nes_df.columns):
            nv = nes_df.iloc[i, j]
            fv = fdr_df.loc[term, col] if term in fdr_df.index and col in fdr_df.columns else 1
            s = '***' if fv < 0.01 else '**' if fv < 0.05 else '*' if fv < 0.25 else ''
            annot_arr[i, j] = f'{nv:.1f}{s}'

    fig, ax = plt.subplots(figsize=(14, max(8, len(nes_df) * 0.32)))
    cmap = LinearSegmentedColormap.from_list(
        '', ['#2166AC', '#67A9CF', '#F7F7F7', '#EF8A62', '#B2182B'])
    vm = np.percentile(np.abs(nes_df.values), 95)

    sns.heatmap(nes_df, cmap=cmap, center=0, vmin=-vm, vmax=vm, ax=ax,
                linewidths=0.5, annot=annot_arr, fmt='',
                annot_kws={'fontsize': 6.5},
                cbar_kws={'label': 'NES', 'shrink': 0.4})
    ax.set_title('MSigDB Hallmark Fingerprint across Drug Clusters\n'
                 '(*** FDR<0.01  ** FDR<0.05  * FDR<0.25)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Drug Cluster', fontsize=11)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=8.5)
    ax.tick_params(axis='x', labelsize=10, rotation=0)
    plt.tight_layout()
    save_fig("06_hallmark_fingerprint")
    plt.close()

    print("\n  Cluster Fingerprints (top Hallmark):")
    for col, fp in fingerprint.items():
        ar = 'UP' if fp['NES'] > 0 else 'DN'
        print(f"    {col}: {fp['term']} {ar} (NES={fp['NES']:+.2f})")


# =============================================================
#                        MAIN
# =============================================================
if __name__ == "__main__":
    T0 = time.time()
    print("=" * 70)
    print("  CLUSTER BIOLOGY ANNOTATION PIPELINE (v2 FIXED)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---------- Read raw data ----------
    data_raw = pd.read_csv(DATA_FILE, delimiter=',', index_col=0)
    print(f"\nRaw Data: {data_raw.shape[0]} drugs x {data_raw.shape[1]} proteins")
    print(f"Column examples: {data_raw.columns[:5].tolist()}")

    # ---------- STEP 0: ID mapping ----------
    mapping = load_local_mapping(MAPPING_FILE)
    data = convert_data_columns(data_raw, mapping)
    data.to_csv(os.path.join(OUTPUT_DIR, "00_data_gene_symbols.csv"))
    print(f"\n  Converted data saved")

    # ---------- Read clustering labels ----------
    df_labels = pd.read_csv(LABEL_FILE)
    df_labels = df_labels.set_index('Drug')
    labels = df_labels.loc[data.index, 'Cluster'].values
    drug_names = data.index.tolist()
    print(f"\nClusters: {dict(sorted(Counter(labels).items()))}")

    # ---------- STEP 1 ----------
    all_signatures = extract_signature_proteins(data, labels)

    # ---------- STEP 2 ----------
    all_enrichment = pathway_enrichment(
        all_signatures, background_genes=data.columns.tolist())

    # ---------- STEP 2b ----------
    gsea_results, gsea_by_library = gsea_prerank_analysis(all_signatures)

    # ---------- STEP 3 ----------
    plot_signature_heatmap(data, labels, all_signatures)

    # ---------- STEP 4 ----------
    plot_enrichment_summary(all_enrichment, gsea_results, gsea_by_library, labels)

    # ---------- STEP 5 ----------
    plot_cluster_profile_heatmap(data, labels)

    # ---------- STEP 6 ----------
    naming = generate_biology_report(
        all_signatures, all_enrichment, gsea_results,
        gsea_by_library, labels, drug_names)

    # ---------- STEP 7 ----------
    plot_hallmark_fingerprint(gsea_results, labels)

    # ---------- Complete ----------
    elapsed = time.time() - T0
    print(f"\n{'=' * 70}")
    print(f"  DONE! Total time: {elapsed / 60:.1f} minutes")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
