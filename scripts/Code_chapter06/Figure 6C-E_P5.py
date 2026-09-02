"""
═══════════════════════════════════════════════════════════════════════════════
  Integrated Analysis: TCM Proteomics → DRP Projection + CPS6 Scoring

  Input: TCM protein expression matrix
  Outputs:
    1. DRP projection cluster + Confidence Score
    2. Top 10 drugs ranked by cosine similarity
    3. CPS percentages (probability scores for the 10 DRPs)
    4. ✅ CPS6 score + percentile (cell-death mechanism)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import percentileofscore
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration — Only modify the paths in this section                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Reference data paths (fixed)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter06"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter06"
REF_MATRIX_PATH = DATA_ROOT / "01_HepG2_367_Drugs_log2FC_6682_for_UMAP.csv"
UMAP_COORD_PATH = DATA_ROOT / "HepG2_coordinate system_Cluster.csv"
LABEL_PATH = DATA_ROOT / "HepG2_label.csv"

# TCM data path (user input)
TCM_MATRIX_PATH = DATA_ROOT / "TCM_24_combat_matrix.csv"

# Output path
OUTPUT_DIR = RESULTS_ROOT / "05_DRP_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
K_NN = 10  # Number of Top-K neighbors

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DRP Definitions                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DRP_NAMES = {
    1: "E2F/MYC cell-cycle + hepatocyte suppression",
    2: "Hepatocyte metabolic/detox activation",
    3: "E2F/G2M cycling + metabolic downshift",
    4: "NFκB/STAT3 inflammatory + EMT/angiogenesis",
    5: "Genotoxic/p53-apoptosis + IFN",
    6: "Minimal proteomic remodeling",
    7: "Moderate cycling + low inflammatory",
    8: "Microtubule/mitotic stress + cytokine-EMT",
    9: "Global pathway repression + residual cycling",
    10: "HDAC inhibitor-like epigenetic reprogramming",
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ✅ CPS6 Model Definition                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

CPS6_WEIGHTS = {
    'NAPA': 20.57534422,
    'PLEKHA7': -11.15330576,
    'TBL1X': -20.98158776,
    'ACSL1': 20.77593761,
    'ARL8B': 19.05640755,
    'CCDC90B': 13.69596005
}

CPS6_INTERCEPT = 10.5291701954019

CPS6_INFO = {
    'NAPA': {'uniprot': 'P54920', 'module': 'Intracellular_Transport'},
    'PLEKHA7': {'uniprot': 'Q6IQ23', 'module': 'Cell_Adhesion'},
    'TBL1X': {'uniprot': 'O60907', 'module': 'Transcription_Regulation'},
    'ACSL1': {'uniprot': 'P33121', 'module': 'Lipid_metabolism'},
    'ARL8B': {'uniprot': 'Q9NVJ2', 'module': 'Intracellular_Transport'},
    'CCDC90B': {'uniprot': 'Q9GZT6', 'module': 'Mitochondrion'},
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helper Functions                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def find_protein(cols, gene, uid):
    """Perform approximate protein matching against column names."""
    cols_list = list(cols)

    # Direct matching
    if gene in cols_list:
        return gene
    if uid in cols_list:
        return uid

    # Approximate matching (case-insensitive)
    for c in cols_list:
        cu = c.upper()
        if gene.upper() in cu or uid.upper() in cu:
            return c

    return None


def compute_CPS6_score(df, gene_mapping):
    """Calculate the CPS6 score."""
    score = pd.Series(CPS6_INTERCEPT, index=df.index, dtype=float)

    for gene, col_name in gene_mapping.items():
        if col_name and col_name in df.columns:
            score += CPS6_WEIGHTS[gene] * df[col_name]

    return score


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: Load Data                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 1: LOAD DATA")
print("=" * 80)

# 1.1 Load reference data
print("  Loading reference data...")
ref_data = pd.read_csv(REF_MATRIX_PATH, index_col=0)
print(f"    Reference matrix: {ref_data.shape[0]} drugs × {ref_data.shape[1]} proteins")

# 1.2 Load UMAP coordinates and labels
umap_df = pd.read_csv(UMAP_COORD_PATH)
label_df = pd.read_csv(LABEL_PATH)

# Construct reference information table
id_col = umap_df.columns[0]
u1_col = [c for c in umap_df.columns if 'umap1' in c.lower() or 'umap_1' in c.lower()][0]
u2_col = [c for c in umap_df.columns if 'umap2' in c.lower() or 'umap_2' in c.lower()][0]
cl_col = [c for c in label_df.columns if any(k in c.lower() for k in ['cluster', 'label'])][0]

ref_info = umap_df[[id_col, u1_col, u2_col]].merge(
    label_df[[label_df.columns[0], cl_col]],
    left_on=id_col, right_on=label_df.columns[0], how='inner'
)
ref_info = ref_info.rename(columns={id_col: 'Drug', u1_col: 'UMAP1', u2_col: 'UMAP2', cl_col: 'Cluster'})
ref_info = ref_info[['Drug', 'UMAP1', 'UMAP2', 'Cluster']].copy()
ref_info['Cluster'] = ref_info['Cluster'].astype(int)

print(f"  Reference information table: {len(ref_info)} drugs")

# 1.3 Load TCM data
print("  Loading TCM data...")
if TCM_MATRIX_PATH.endswith('.xlsx'):
    tcm_data = pd.read_excel(TCM_MATRIX_PATH, index_col=0)
else:
    tcm_data = pd.read_csv(TCM_MATRIX_PATH, index_col=0)

print(f"    Raw TCM matrix: {tcm_data.shape[0]} × {tcm_data.shape[1]}")

# Check whether transposition is required
if tcm_data.shape[0] > tcm_data.shape[1]:
    print(f"  ⚠️ Detected rows ({tcm_data.shape[0]}) > columns ({tcm_data.shape[1]}); transposing matrix")
    tcm_data = tcm_data.T

tcm_data = tcm_data.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"    Final TCM matrix: {tcm_data.shape[0]} compounds × {tcm_data.shape[1]} proteins")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: Data Alignment                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 2: DATA ALIGNMENT")
print("=" * 80)

# Identify shared proteins for DRP analysis
common_proteins = sorted(set(ref_data.columns) & set(tcm_data.columns))

if len(common_proteins) == 0:
    print("\n❌ ERROR: No shared proteins were found!")
    print(f"  First 5 proteins in reference data: {list(ref_data.columns[:5])}")
    print(f"  First 5 proteins in TCM data:       {list(tcm_data.columns[:5])}")
    exit(1)

print(f"  Shared proteins: {len(common_proteins)} ({len(common_proteins) / len(ref_data.columns) * 100:.1f}%)")

# Align datasets
ref_aligned = ref_data[common_proteins]
tcm_aligned = tcm_data[common_proteins]

# Ensure that all reference drugs are present in ref_info
common_drugs = sorted(set(ref_aligned.index) & set(ref_info['Drug']))
ref_aligned = ref_aligned.loc[common_drugs]
ref_info = ref_info[ref_info['Drug'].isin(common_drugs)].reset_index(drop=True)

print(f"  Reference drugs: {len(ref_aligned)}")
print(f"  TCM compounds: {len(tcm_aligned)}")

# Extract compound names
tcm_compound_names = list(tcm_aligned.index)
ref_drug_names = list(ref_aligned.index)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2.5: ✅ CPS6 Protein Matching                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 2.5: CPS6 PROTEIN MATCHING")
print("=" * 80)

# Match CPS6 proteins
ref_cps6_mapping = {}
tcm_cps6_mapping = {}

print(f"\n  {'Gene':<10} {'UniProt':<10} {'REF_Col':<20} {'TCM_Col':<20} {'Status'}")
print(f"  {'-' * 70}")

for gene, info in CPS6_INFO.items():
    uid = info['uniprot']

    ref_col = find_protein(ref_data.columns, gene, uid)
    tcm_col = find_protein(tcm_data.columns, gene, uid)

    ref_cps6_mapping[gene] = ref_col
    tcm_cps6_mapping[gene] = tcm_col

    status = 'OK' if (ref_col and tcm_col) else ('REF_MISS' if not ref_col else 'TCM_MISS')
    print(f"  {gene:<10} {uid:<10} {str(ref_col):<20} {str(tcm_col):<20} {status}")

# Identify available CPS6 proteins
available_cps6_genes = [g for g, col in tcm_cps6_mapping.items() if col is not None]
missing_cps6_genes = [g for g in CPS6_WEIGHTS if g not in available_cps6_genes]

print(f"\n  ✅ CPS6 matched: {len(available_cps6_genes)}/6 proteins")
if missing_cps6_genes:
    print(f"  ⚠️ Missing: {missing_cps6_genes}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: Standardization                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 3: DATA STANDARDIZATION")
print("=" * 80)

scaler = StandardScaler()
ref_std = scaler.fit_transform(ref_aligned.values)
tcm_std = scaler.transform(tcm_aligned.values)

# Extract labels and coordinates from reference data
ref_labels = ref_info.set_index('Drug').loc[ref_drug_names, 'Cluster'].values
ref_umap1 = ref_info.set_index('Drug').loc[ref_drug_names, 'UMAP1'].values
ref_umap2 = ref_info.set_index('Drug').loc[ref_drug_names, 'UMAP2'].values

print("  ✓ Standardization completed")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: Similarity Calculation and DRP Projection                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 4: DRP PROJECTION ANALYSIS")
print("=" * 80)

# Calculate cosine similarity
print(f"  Calculating cosine similarity matrix...")
cos_sim = cosine_similarity(tcm_aligned.values, ref_aligned.values)
print(f"    Matrix shape: {cos_sim.shape}")

# Perform DRP analysis for each TCM compound
results = []

print(f"\n  Analyzing {len(tcm_compound_names)} compounds:")

for i, compound in enumerate(tcm_compound_names):
    print(f"    [{i + 1}/{len(tcm_compound_names)}] {compound}...", end=" ")

    # Retrieve similarities between this compound and all reference drugs
    similarities = cos_sim[i]

    # Identify the Top-K most similar reference drugs
    top_k_indices = np.argsort(similarities)[::-1][:K_NN]

    # Extract Top-K information
    top_k_drugs = [ref_drug_names[idx] for idx in top_k_indices]
    top_k_sims = similarities[top_k_indices]
    top_k_clusters = ref_labels[top_k_indices]

    # Calculate weights using similarity values
    weights = np.maximum(top_k_sims, 0)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(K_NN) / K_NN

    # Perform weighted voting across the 10 DRP clusters
    cluster_scores = {}
    for cluster, weight in zip(top_k_clusters, weights):
        cluster_scores[cluster] = cluster_scores.get(cluster, 0) + weight

    # Determine the best-matching DRP cluster
    assigned_drp = max(cluster_scores, key=cluster_scores.get)
    confidence = cluster_scores[assigned_drp]

    print(f"C{assigned_drp} ({confidence * 100:.1f}%)")

    # Construct result row
    result = {
        'Compound_ID': compound,
        'Assigned_DRP': f"C{assigned_drp}",
        'DRP_Description': DRP_NAMES[assigned_drp],
        'Confidence_Score_%': round(confidence * 100, 2),
    }

    # Add probability scores (CPS) for all 10 DRPs
    for drp in range(1, 11):
        result[f'DRP_C{drp}_%'] = round(cluster_scores.get(drp, 0) * 100, 2)

    # Add Top-10 similar reference drugs
    for rank in range(10):
        idx = np.argsort(similarities)[::-1][rank]
        result[f'Top{rank + 1}_Drug'] = ref_drug_names[idx]
        result[f'Top{rank + 1}_Similarity'] = round(similarities[idx], 4)
        result[f'Top{rank + 1}_DRP'] = f"C{ref_labels[idx]}"

    results.append(result)

# Convert to DataFrame
df_results = pd.DataFrame(results)

print(f"\n  ✓ DRP analysis completed")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: ✅ CPS6 Score Calculation                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 5: CPS6 SCORE CALCULATION")
print("=" * 80)

# 5.1 Calculate CPS6 scores for reference drugs for percentile comparison
print("  Calculating CPS6 scores for reference drugs...")
ref_cps6_scores = compute_CPS6_score(ref_data, ref_cps6_mapping)
print(f"    Reference CPS6 range: [{ref_cps6_scores.min():.1f}, {ref_cps6_scores.max():.1f}]")
print(f"    Reference CPS6 mean: {ref_cps6_scores.mean():.2f} ± {ref_cps6_scores.std():.2f}")

# 5.2 Calculate CPS6 scores for TCM compounds
print("  Calculating CPS6 scores for TCM compounds...")
tcm_cps6_scores = compute_CPS6_score(tcm_data, tcm_cps6_mapping)
print(f"    TCM CPS6 range: [{tcm_cps6_scores.min():.1f}, {tcm_cps6_scores.max():.1f}]")
print(f"    TCM CPS6 mean: {tcm_cps6_scores.mean():.2f} ± {tcm_cps6_scores.std():.2f}")

# 5.3 Calculate percentiles
print("\n  Calculating CPS6 percentiles...")
tcm_cps6_percentiles = {}

for compound in tcm_compound_names:
    score = float(tcm_cps6_scores.loc[compound])
    percentile = percentileofscore(ref_cps6_scores.values, score)
    tcm_cps6_percentiles[compound] = percentile

print(f"    Mean percentile: {np.mean(list(tcm_cps6_percentiles.values())):.1f}")
print(f"    >90th percentile: {sum(1 for p in tcm_cps6_percentiles.values() if p > 90)} compounds")
print(f"    <10th percentile: {sum(1 for p in tcm_cps6_percentiles.values() if p < 10)} compounds")

# 5.4 Add CPS6 information to the result table
df_results['CPS6_Score'] = [round(float(tcm_cps6_scores.loc[cpd]), 2) for cpd in tcm_compound_names]
df_results['CPS6_Percentile'] = [round(tcm_cps6_percentiles[cpd], 1) for cpd in tcm_compound_names]

# Add CPS6 tiers
df_results['CPS6_Tier'] = df_results['CPS6_Percentile'].apply(
    lambda p: 'High (>75th)' if p > 75 else ('Low (<25th)' if p < 25 else 'Medium')
)

# 5.5 Add the contribution of each CPS6 protein
for gene in available_cps6_genes:
    col_name = tcm_cps6_mapping[gene]
    if col_name:
        df_results[f'CPS6_{gene}_log2FC'] = [round(float(tcm_data.loc[cpd, col_name]), 3)
                                             for cpd in tcm_compound_names]
        df_results[f'CPS6_{gene}_contribution'] = [
            round(CPS6_WEIGHTS[gene] * float(tcm_data.loc[cpd, col_name]), 2)
            for cpd in tcm_compound_names
        ]

print(f"\n  ✓ CPS6 analysis completed")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: Save Results                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("  STEP 6: SAVE RESULTS")
print("=" * 80)

# Prepare output Excel file
output_file = os.path.join(OUTPUT_DIR, "TCM_DRP_CPS6_Analysis_Results.xlsx")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # ═══════════════════════════════════════════════════════════
    # Sheet 1: Projection summary including CPS6
    # ═══════════════════════════════════════════════════════════

    summary_cols = ['Compound_ID', 'Assigned_DRP', 'DRP_Description', 'Confidence_Score_%',
                    'CPS6_Score', 'CPS6_Percentile', 'CPS6_Tier']
    df_summary = df_results[summary_cols].copy()
    df_summary.to_excel(writer, sheet_name='1_Projection_Summary', index=False)

    # ═══════════════════════════════════════════════════════════
    # Sheet 2: DRP CPS percentages
    # ═══════════════════════════════════════════════════════════

    cps_cols = ['Compound_ID'] + [f'DRP_C{i}_%' for i in range(1, 11)]
    df_cps = df_results[cps_cols].copy()
    df_cps.to_excel(writer, sheet_name='2_DRP_CPS_Percentages', index=False)

    # ═══════════════════════════════════════════════════════════
    # Sheet 3: ✅ Detailed CPS6 scores
    # ═══════════════════════════════════════════════════════════

    cps6_cols = ['Compound_ID', 'CPS6_Score', 'CPS6_Percentile', 'CPS6_Tier']

    # Add information for each CPS6 protein
    for gene in available_cps6_genes:
        cps6_cols.extend([f'CPS6_{gene}_log2FC', f'CPS6_{gene}_contribution'])

    df_cps6 = df_results[cps6_cols].copy()
    df_cps6 = df_cps6.sort_values('CPS6_Score', ascending=False)
    df_cps6.to_excel(writer, sheet_name='3_CPS6_Detailed_Scores', index=False)

    # ═══════════════════════════════════════════════════════════
    # Sheet 4: Top 10 similar drugs
    # ═══════════════════════════════════════════════════════════

    top10_data = []
    for _, row in df_results.iterrows():
        for rank in range(1, 11):
            top10_data.append({
                'Compound_ID': row['Compound_ID'],
                'Rank': rank,
                'Reference_Drug': row[f'Top{rank}_Drug'],
                'Cosine_Similarity': row[f'Top{rank}_Similarity'],
                'Ref_DRP': row[f'Top{rank}_DRP'],
            })

    df_top10 = pd.DataFrame(top10_data)
    df_top10.to_excel(writer, sheet_name='4_Top10_Similar_Drugs', index=False)

    # ═══════════════════════════════════════════════════════════
    # Sheet 5: Complete results
    # ═══════════════════════════════════════════════════════════

    df_results.to_excel(writer, sheet_name='5_Complete_Results', index=False)


# Also save individual CSV files
df_summary.to_csv(os.path.join(OUTPUT_DIR, "1_Projection_Summary.csv"), index=False, encoding='utf-8-sig')
df_cps.to_csv(os.path.join(OUTPUT_DIR, "2_DRP_CPS_Percentages.csv"), index=False, encoding='utf-8-sig')
df_cps6.to_csv(os.path.join(OUTPUT_DIR, "3_CPS6_Detailed_Scores.csv"), index=False, encoding='utf-8-sig')
df_top10.to_csv(os.path.join(OUTPUT_DIR, "4_Top10_Similar_Drugs.csv"), index=False, encoding='utf-8-sig')

print(f"  ✓ CSV files saved")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Final Report                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "╔" + "=" * 78 + "╗")
print("║" + " ANALYSIS COMPLETED!".center(78) + "║")
print("╠" + "=" * 78 + "╣")
print(f"║  Compounds analyzed: {len(tcm_compound_names)}".ljust(79) + "║")
print(f"║  Reference drugs: {len(ref_drug_names)}".ljust(79) + "║")
print(f"║  Shared proteins: {len(common_proteins)}".ljust(79) + "║")
print(f"║  CPS6 proteins matched: {len(available_cps6_genes)}/6".ljust(79) + "║")
print("╠" + "=" * 78 + "╣")
print("║  Output file:".ljust(79) + "║")
print(f"║    {output_file}".ljust(79) + "║")
print("╠" + "=" * 78 + "╣")
print("║  Excel worksheets:".ljust(79) + "║")
print("║    1_Projection_Summary       - Integrated DRP + CPS6 summary".ljust(79) + "║")
print("║    2_DRP_CPS_Percentages      - Probability scores for 10 DRPs".ljust(79) + "║")
print("║    3_CPS6_Detailed_Scores     - CPS6 scores, percentiles, and contributions".ljust(79) + "║")
print("║    4_Top10_Similar_Drugs      - Most similar reference drugs".ljust(79) + "║")
print("║    5_Complete_Results         - Complete analysis results".ljust(79) + "║")
print("╠" + "=" * 78 + "╣")
print("║  Result preview:".ljust(79) + "║")

for _, row in df_summary.head(5).iterrows():
    line = (f"║    {row['Compound_ID']:10s} → DRP:{row['Assigned_DRP']:3s} ({row['Confidence_Score_%']:4.1f}%) | "
            f"CPS6:{row['CPS6_Score']:6.1f} (P{row['CPS6_Percentile']:4.1f})")
    print(line.ljust(79) + "║")

if len(df_summary) > 5:
    print("║    ...".ljust(79) + "║")

print("╠" + "=" * 78 + "╣")
print("║  CPS6 score statistics:".ljust(79) + "║")

# CPS6 tier statistics
high_cps6 = df_results[df_results['CPS6_Tier'] == 'High (>75th)']
medium_cps6 = df_results[df_results['CPS6_Tier'] == 'Medium']
low_cps6 = df_results[df_results['CPS6_Tier'] == 'Low (<25th)']

print(f"║    High tendency (>75th): {len(high_cps6)} compounds".ljust(79) + "║")
print(f"║    Moderate tendency:     {len(medium_cps6)} compounds".ljust(79) + "║")
print(f"║    Low tendency (<25th):  {len(low_cps6)} compounds".ljust(79) + "║")

# Display extreme compounds
if len(high_cps6) > 0:
    top_cps6 = df_results.nlargest(1, 'CPS6_Score').iloc[0]
    print(
        f"║    Highest CPS6: {top_cps6['Compound_ID']} ({top_cps6['CPS6_Score']:.1f}, P{top_cps6['CPS6_Percentile']:.0f})".ljust(
            79) + "║")

if len(low_cps6) > 0:
    bottom_cps6 = df_results.nsmallest(1, 'CPS6_Score').iloc[0]
    print(
        f"║    Lowest CPS6: {bottom_cps6['Compound_ID']} ({bottom_cps6['CPS6_Score']:.1f}, P{bottom_cps6['CPS6_Percentile']:.0f})".ljust(
            79) + "║")

print("╚" + "=" * 78 + "╝")

print(f"\n🎉 Analysis completed! Results are available at: {OUTPUT_DIR}\n")

# ═══════════════════════════════════════════════════════════════════════════
#  ✅ Additional Output: CPS6 Tier Comparison Table
# ═══════════════════════════════════════════════════════════════════════════

cps6_tier_summary = pd.DataFrame({
    'CPS6_Tier': ['High (>75th)', 'Medium', 'Low (<25th)'],
    'N_Compounds': [len(high_cps6), len(medium_cps6), len(low_cps6)],
    'CPS6_Range': [
        f"[{high_cps6['CPS6_Score'].min():.1f}, {high_cps6['CPS6_Score'].max():.1f}]" if len(high_cps6) > 0 else "[]",
        f"[{medium_cps6['CPS6_Score'].min():.1f}, {medium_cps6['CPS6_Score'].max():.1f}]" if len(
            medium_cps6) > 0 else "[]",
        f"[{low_cps6['CPS6_Score'].min():.1f}, {low_cps6['CPS6_Score'].max():.1f}]" if len(low_cps6) > 0 else "[]"
    ],
    'Examples': [
        ', '.join(high_cps6['Compound_ID'].head(3).tolist()) if len(high_cps6) > 0 else "",
        ', '.join(medium_cps6['Compound_ID'].head(3).tolist()) if len(medium_cps6) > 0 else "",
        ', '.join(low_cps6['Compound_ID'].head(3).tolist()) if len(low_cps6) > 0 else ""
    ]
})

cps6_tier_summary.to_csv(os.path.join(OUTPUT_DIR, "CPS6_Tier_Summary.csv"),
                         index=False, encoding='utf-8-sig')

print(f"✅ CPS6 tier summary generated: CPS6_Tier_Summary.csv\n")

# ═══════════════════════════════════════════════════════════════════════════
#  ✅ Generate Interactive HTML Report
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  GENERATING HTML REPORT...")
print("=" * 80)

# Prepare data
html_data = []
for _, row in df_results.iterrows():
    # Prepare Top-10 data
    top10 = []
    for rank in range(1, 11):
        top10.append({
            'drug': row[f'Top{rank}_Drug'],
            'similarity': float(row[f'Top{rank}_Similarity']),
            'drp': row[f'Top{rank}_DRP']
        })

    html_data.append({
        'id': row['Compound_ID'],
        'drp': row['Assigned_DRP'],
        'drp_desc': row['DRP_Description'],
        'confidence': float(row['Confidence_Score_%']),
        'cps6_score': float(row['CPS6_Score']),
        'cps6_percentile': float(row['CPS6_Percentile']),
        'cps6_tier': row['CPS6_Tier'],
        'top10': top10
    })

# Generate HTML
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drug Response Phenotype Projection Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .selector-container {
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }

        .selector-container label {
            font-size: 1.2em;
            font-weight: 600;
            color: #495057;
            display: block;
            margin-bottom: 10px;
        }

        #compoundSelect {
            width: 100%;
            padding: 15px 20px;
            font-size: 1.1em;
            border: 2px solid #ced4da;
            border-radius: 10px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
        }

        #compoundSelect:hover {
            border-color: #667eea;
        }

        #compoundSelect:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .content {
            padding: 40px;
        }

        .modules {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .module {
            background: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .module:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }

        .module-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .info-item {
            margin: 15px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .info-label {
            font-weight: 600;
            color: #6c757d;
            font-size: 1em;
        }

        .info-value {
            font-size: 1.3em;
            font-weight: 700;
            color: #212529;
        }

        .drp-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 1.1em;
        }

        .confidence-high {
            color: #28a745;
        }

        .confidence-medium {
            color: #ffc107;
        }

        .confidence-low {
            color: #dc3545;
        }

        .cps6-tier-high {
            background: #dc3545;
            color: white;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
        }

        .cps6-tier-medium {
            background: #ffc107;
            color: #212529;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
        }

        .cps6-tier-low {
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
        }

        /* ✅ Top-10 container styling */
        .top10-container {
            grid-column: 1 / -1;
            background: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .drug-bar-item {
            display: flex;
            align-items: center;
            margin: 12px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 12px;
            transition: all 0.3s;
            border-left: 5px solid #667eea;
        }
        
        .drug-bar-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        
        .bar-rank {
            min-width: 70px;
            font-size: 1.8em;
            font-weight: 700;
            text-align: center;
        }
        
        .bar-rank.top-1 { color: #FFD700; }
        .bar-rank.top-2 { color: #C0C0C0; }
        .bar-rank.top-3 { color: #CD7F32; }
        .bar-rank.other { color: #667eea; }
        
        .bar-info {
            flex: 1;
            margin: 0 20px;
        }
        
        .bar-drug-name {
            font-weight: 600;
            color: #212529;
            margin-bottom: 5px;
            font-size: 1.1em;
        }
        
        .bar-drp {
            font-size: 0.9em;
            color: #6c757d;
            background: #e9ecef;
            padding: 3px 10px;
            border-radius: 10px;
            display: inline-block;
        }
        
        .bar-chart {
            flex: 2;
            position: relative;
            min-width: 300px;
        }
        
        .bar-bg {
            height: 35px;
            background: #e9ecef;
            border-radius: 17px;
            overflow: hidden;
            position: relative;
        }
        
        .bar-fill {
            height: 100%;
            background: linear-gradient(to right, #667eea, #764ba2);
            border-radius: 17px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .bar-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .bar-value {
            min-width: 80px;
            text-align: right;
            font-weight: 700;
            font-size: 1.3em;
            color: #667eea;
        }

        .percentile-bar {
            width: 100%;
            height: 30px;
            background: linear-gradient(to right, #28a745, #ffc107, #dc3545);
            border-radius: 15px;
            position: relative;
            margin-top: 10px;
        }

        .percentile-marker {
            position: absolute;
            top: -5px;
            width: 4px;
            height: 40px;
            background: #212529;
            border-radius: 2px;
        }

        .percentile-marker::after {
            content: attr(data-value);
            position: absolute;
            top: -25px;
            left: 50%;
            transform: translateX(-50%);
            background: #212529;
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.8em;
            white-space: nowrap;
        }

        @media (max-width: 768px) {
            .modules {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 1.8em;
            }

            body {
                padding: 10px;
            }

            .content {
                padding: 20px;
            }
            
            .bar-chart {
                min-width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Drug Response Phenotype (DRP) Projection Report</h1>
            <p>DRP Projection & CPS6 Cell-Death Tendency</p>
        </div>

        <div class="selector-container">
            <label for="compoundSelect">📊 Compound:</label>
            <select id="compoundSelect" onchange="updateDisplay()">
                <option value="">-- Select a Compound --</option>
            </select>
        </div>

        <div class="content">
            <div class="modules">
                <!-- Module 1: Drug Response Phenotype Projection -->
                <div class="module">
                    <div class="module-title">📌 DRP Projection</div>
                    <div class="info-item">
                        <span class="info-label">Assigned DRP</span>
                        <span class="info-value drp-badge" id="drp-value">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">DRP Description</span>
                        <span class="info-value" id="drp-desc" style="font-size: 0.9em; text-align: right;">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Confidence Score</span>
                        <span class="info-value" id="confidence-value">-</span>
                    </div>
                </div>

                <!-- Module 2: CPS6 Score -->
                <div class="module">
                    <div class="module-title">💀 CPS6 Cell-Death Tendency</div>
                    <div class="info-item">
                        <span class="info-label">CPS6 Score</span>
                        <span class="info-value" id="cps6-score">-</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">CPS6 Percentile Relative to the Reference Drug Dataset</span>
                        <span class="info-value" id="cps6-percentile">-</span>
                    </div>
                    <div class="percentile-bar" id="percentile-bar"></div>
                    <div class="info-item" style="margin-top: 20px;">
                        <span class="info-label">Cell-Death Tendency</span>
                        <span class="info-value" id="cps6-tier">-</span>
                    </div>
                </div>

                <!-- ✅ Module 3: Top-10 Similar Drugs (Pure CSS Implementation) -->
                <div class="top10-container">
                    <div class="module-title">🔝 Top 10 Similar Reference Antineoplastic Agents</div>
                    <div id="top10Bars"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const compoundsData = ''' + str(html_data).replace("'", '"') + ''';

        // Populate the drop-down menu
        const selectElement = document.getElementById('compoundSelect');
        compoundsData.forEach((compound, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = compound.id;
            selectElement.appendChild(option);
        });

        // Update display
        function updateDisplay() {
            const selectedIndex = document.getElementById('compoundSelect').value;

            if (selectedIndex === '') {
                return;
            }

            const data = compoundsData[selectedIndex];

            // Update Module 1: DRP
            document.getElementById('drp-value').textContent = data.drp;
            document.getElementById('drp-desc').textContent = data.drp_desc;

            const confidenceValue = data.confidence;
            const confidenceElement = document.getElementById('confidence-value');
            confidenceElement.textContent = confidenceValue.toFixed(2) + '%';

            // Confidence color
            confidenceElement.className = 'info-value';
            if (confidenceValue >= 70) {
                confidenceElement.classList.add('confidence-high');
            } else if (confidenceValue >= 50) {
                confidenceElement.classList.add('confidence-medium');
            } else {
                confidenceElement.classList.add('confidence-low');
            }

            // Update Module 2: CPS6
            document.getElementById('cps6-score').textContent = data.cps6_score.toFixed(2);
            document.getElementById('cps6-percentile').textContent = data.cps6_percentile.toFixed(1) + '%';

            // Update percentile marker
            const percentileBar = document.getElementById('percentile-bar');
            percentileBar.innerHTML = `<div class="percentile-marker" style="left: ${data.cps6_percentile}%;" data-value="${data.cps6_percentile.toFixed(1)}%"></div>`;

            // CPS6 Tier
            const tierElement = document.getElementById('cps6-tier');
            tierElement.textContent = data.cps6_tier;
            tierElement.className = 'info-value';
            if (data.cps6_tier.includes('High')) {
                tierElement.classList.add('cps6-tier-high');
            } else if (data.cps6_tier.includes('Medium')) {
                tierElement.classList.add('cps6-tier-medium');
            } else {
                tierElement.classList.add('cps6-tier-low');
            }

            // Update Module 3: Top-10 bar chart
            updateTop10Bars(data.top10);
        }

        // ✅ Update Top-10 bar chart using pure CSS
        function updateTop10Bars(top10Data) {
            const container = document.getElementById('top10Bars');
            container.innerHTML = '';
            
            const medals = ['🥇', '🥈', '🥉'];
            
            top10Data.forEach((item, index) => {
                const rank = index + 1;
                const similarity = (item.similarity * 100).toFixed(2);
                const rankClass = rank <= 3 ? `top-${rank}` : 'other';
                const rankDisplay = rank <= 3 ? medals[index] : `#${rank}`;
                
                const barItem = document.createElement('div');
                barItem.className = 'drug-bar-item';
                barItem.innerHTML = `
                    <div class="bar-rank ${rankClass}">${rankDisplay}</div>
                    <div class="bar-info">
                        <div class="bar-drug-name">${item.drug}</div>
                        <div class="bar-drp">DRP: ${item.drp}</div>
                    </div>
                    <div class="bar-chart">
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: 0%;" data-width="${similarity}"></div>
                        </div>
                    </div>
                    <div class="bar-value">${similarity}%</div>
                `;
                container.appendChild(barItem);
                
                // Animation effect
                setTimeout(() => {
                    const fill = barItem.querySelector('.bar-fill');
                    fill.style.width = fill.dataset.width + '%';
                }, 100 * index);
            });
        }

        // Select the first compound by default
        if (compoundsData.length > 0) {
            selectElement.selectedIndex = 1;
            updateDisplay();
        }
    </script>
</body>
</html>
'''

# Save HTML file
html_output = os.path.join(OUTPUT_DIR, "TCM_Analysis_Report.html")
with open(html_output, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"  ✅ HTML report generated: {html_output}")
print(f"  📂 Double-click the file to open it in your web browser")
