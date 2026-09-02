"""
═══════════════════════════════════════════════════════════════════════════════
  Batch Processing: In-Depth DRP Projection Analysis
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe
from matplotlib.patches import Circle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import pearsonr, chi2_contingency

# Compatibility fix: support different SciPy versions
try:
    from scipy.stats import binomtest  # Newer SciPy versions (>= 1.7.0)
except ImportError:
    from scipy.stats import binom_test as binomtest  # Older SciPy versions

from scipy.spatial.distance import cdist
from umap import UMAP
from collections import Counter
import os
import glob
import time
import warnings
from datetime import datetime
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter06"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter06"
REF_MATRIX_PATH = DATA_ROOT / "01_HepG2_367_Drugs_log2FC_6682_for_UMAP.csv"
UMAP_COORD_PATH = DATA_ROOT / "HepG2_coordinate system.csv"
LABEL_PATH = DATA_ROOT / "HepG2_label.csv"

DRUG_PROFILES_ROOT = DATA_ROOT / "drug_profiles"
BATCH_OUTPUT_ROOT = RESULTS_ROOT / "03_External perturbatione performance"

K_NN = 3

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

DRP_FAMILY_DICT = {
    1: "I", 3: "I", 9: "I",
    6: "II", 7: "II",
    2: "III", 5: "III", 10: "III",
    4: "IV", 8: "IV",
}

DRP_FAMILY_FULL_NAMES = {
    "I": "I: Cycling/Suppressed (C1,C3,C9)",
    "II": "II: Low-remodeling (C6,C7)",
    "III": "III: DDR-Metabolic-Epigenetic (C2,C5,C10)",
    "IV": "IV: Inflammatory-EMT (C4,C8)",
}

DRP_COLORS = {
    1: "#E41A1C", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#FF7F00",
    6: "#A65628", 7: "#F781BF", 8: "#808080", 9: "#66C2A5", 10: "#E6AB02",
}

FAMILY_COLORS = {
    "I": "#E74C3C", "II": "#95A5A6", "III": "#3498DB", "IV": "#9B59B6",
}

CELLLINE_MARKERS = ['*', 'o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>', 'p', 'H']
CELLLINE_COLORS = plt.cm.Set1(np.linspace(0, 1, 13))


def get_drp_family(drp_cluster):
    """Return the DRP family associated with a given DRP cluster."""
    return DRP_FAMILY_DICT.get(int(drp_cluster), "Unknown")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  In-Depth Analysis Functions                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def calculate_random_baselines():
    """Calculate random-baseline accuracies."""
    num_drp_clusters = len(set(DRP_FAMILY_DICT.keys()))
    num_families = len(set(DRP_FAMILY_DICT.values()))

    drp_baseline = 1.0 / num_drp_clusters
    family_baseline = 1.0 / num_families

    return {
        'drp_baseline': drp_baseline,
        'family_baseline': family_baseline,
        'num_drp_clusters': num_drp_clusters,
        'num_families': num_families
    }


def perform_statistical_test(observed_accuracy, baseline_accuracy, n_samples):
    """
    Perform a binomial test to determine whether the observed accuracy
    is significantly greater than the random baseline.
    """
    n_correct = int(observed_accuracy * n_samples)

    # Handle different SciPy versions
    try:
        # Newer SciPy versions (>= 1.7.0)
        result = binomtest(n_correct, n_samples, baseline_accuracy, alternative='greater')
        p_value = result.pvalue
    except TypeError:
        # Older SciPy versions
        p_value = binomtest(n_correct, n_samples, baseline_accuracy, alternative='greater')

    return {
        'n_samples': n_samples,
        'n_correct': n_correct,
        'observed_accuracy': observed_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'fold_improvement': observed_accuracy / baseline_accuracy if baseline_accuracy > 0 else np.inf
    }


def analyze_transferability_by_drp(results_df):
    """
    Analyze transferability for each reference DRP cluster.
    """
    if 'Reference_DRP' not in results_df.columns or 'Assigned_DRP' not in results_df.columns:
        return None

    transferability = []

    for ref_drp in sorted(results_df['Reference_DRP'].dropna().unique()):
        subset = results_df[results_df['Reference_DRP'] == ref_drp]

        if len(subset) == 0:
            continue

        n_samples = len(subset)
        n_correct = subset['DRP_Match'].sum()
        accuracy = n_correct / n_samples if n_samples > 0 else 0

        avg_confidence = subset['Confidence'].mean()
        avg_distance = subset['Distance_to_Reference'].mean()
        family_match_rate = subset['Family_Match'].mean() if 'Family_Match' in subset.columns else 0

        transferability.append({
            'Reference_DRP': int(ref_drp),
            'DRP_Name': DRP_NAMES.get(int(ref_drp), "Unknown"),
            'Family': get_drp_family(ref_drp),
            'N_Samples': n_samples,
            'DRP_Accuracy': accuracy,
            'Family_Accuracy': family_match_rate,
            'Avg_Confidence': avg_confidence,
            'Avg_Distance': avg_distance,
            'Transferable': accuracy > 0.5
        })

    return pd.DataFrame(transferability).sort_values('DRP_Accuracy', ascending=False)


def analyze_failure_patterns(results_df):
    """
    Analyze patterns associated with failed DRP projections.
    """
    if 'DRP_Match' not in results_df.columns:
        return None

    failure_analysis = {
        'by_cellline': {},
        'by_protein_coverage': {},
        'by_confidence': {},
        'by_reference_drp': {}
    }

    # 1. Analysis by cell line
    for cellline in results_df['Cellline'].unique():
        subset = results_df[results_df['Cellline'] == cellline]
        failure_rate = 1 - subset['DRP_Match'].mean()
        avg_confidence = subset['Confidence'].mean()
        avg_coverage = subset['Common_Proteins'].mean()

        failure_analysis['by_cellline'][cellline] = {
            'failure_rate': failure_rate,
            'n_samples': len(subset),
            'avg_confidence': avg_confidence,
            'avg_coverage': avg_coverage
        }

    # 2. Analysis by protein coverage
    results_df['Coverage_Bin'] = pd.cut(
        results_df['Common_Proteins'],
        bins=[0, 3000, 4000, 5000, 10000],
        labels=['<3000', '3000-4000', '4000-5000', '>5000']
    )

    for coverage_bin in results_df['Coverage_Bin'].dropna().unique():
        subset = results_df[results_df['Coverage_Bin'] == coverage_bin]
        failure_rate = 1 - subset['DRP_Match'].mean()

        failure_analysis['by_protein_coverage'][str(coverage_bin)] = {
            'failure_rate': failure_rate,
            'n_samples': len(subset)
        }

    # 3. Analysis by confidence level
    results_df['Confidence_Bin'] = pd.cut(
        results_df['Confidence'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low<0.3', 'Med0.3-0.5', 'High0.5-0.7', 'VeryHigh>0.7']
    )

    for conf_bin in results_df['Confidence_Bin'].dropna().unique():
        subset = results_df[results_df['Confidence_Bin'] == conf_bin]
        failure_rate = 1 - subset['DRP_Match'].mean()

        failure_analysis['by_confidence'][str(conf_bin)] = {
            'failure_rate': failure_rate,
            'n_samples': len(subset),
            'accuracy': subset['DRP_Match'].mean()
        }

    # 4. Analysis by reference DRP
    if 'Reference_DRP' in results_df.columns:
        for ref_drp in results_df['Reference_DRP'].dropna().unique():
            subset = results_df[results_df['Reference_DRP'] == ref_drp]
            failure_rate = 1 - subset['DRP_Match'].mean()

            failure_analysis['by_reference_drp'][int(ref_drp)] = {
                'failure_rate': failure_rate,
                'n_samples': len(subset)
            }

    return failure_analysis


def analyze_confidence_calibration(results_df):
    """
    Analyze the calibration relationship between confidence and accuracy.
    """
    if 'Confidence' not in results_df.columns or 'DRP_Match' not in results_df.columns:
        return None

    # Divide confidence scores into quantile-based bins
    n_bins = 10
    results_df['Conf_Decile'] = pd.qcut(
        results_df['Confidence'],
        q=n_bins,
        labels=False,
        duplicates='drop'
    )

    calibration_data = []

    for decile in sorted(results_df['Conf_Decile'].dropna().unique()):
        subset = results_df[results_df['Conf_Decile'] == decile]

        avg_confidence = subset['Confidence'].mean()
        avg_accuracy = subset['DRP_Match'].mean()
        n_samples = len(subset)

        calibration_data.append({
            'decile': decile,
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'n_samples': n_samples,
            'calibration_error': abs(avg_confidence - avg_accuracy)
        })

    calibration_df = pd.DataFrame(calibration_data)

    # Expected Calibration Error (ECE)
    ece = np.average(
        calibration_df['calibration_error'],
        weights=calibration_df['n_samples']
    )

    return {
        'calibration_df': calibration_df,
        'ece': ece
    }


def recommend_confidence_threshold(results_df, target_precision=0.8):
    """
    Recommend a confidence threshold to achieve the target precision.
    """
    if 'Confidence' not in results_df.columns or 'DRP_Match' not in results_df.columns:
        return None

    sorted_df = results_df.sort_values('Confidence', ascending=False).copy()
    sorted_df['Cumulative_Accuracy'] = sorted_df['DRP_Match'].expanding().mean()
    sorted_df['Cumulative_Count'] = range(1, len(sorted_df) + 1)

    candidates = sorted_df[sorted_df['Cumulative_Accuracy'] >= target_precision]

    if len(candidates) == 0:
        return {
            'target_precision': target_precision,
            'achievable': False,
            'max_precision': sorted_df['Cumulative_Accuracy'].max(),
            'recommendation': 'Target precision not achievable'
        }

    threshold_idx = candidates.index[-1]
    recommended_threshold = sorted_df.loc[threshold_idx, 'Confidence']
    n_retained = len(candidates)
    retention_rate = n_retained / len(sorted_df)

    return {
        'target_precision': target_precision,
        'achievable': True,
        'recommended_threshold': recommended_threshold,
        'n_retained': n_retained,
        'retention_rate': retention_rate,
        'actual_precision': sorted_df.loc[threshold_idx, 'Cumulative_Accuracy']
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  process_single_drug Function (Complete Previous Version)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def process_single_drug(drug_name, drug_folder, output_dir, ref_data, umap_df, label_df, ref_info):
    """
    Process the complete analysis workflow for a single drug.
    This function uses the complete implementation from the previous version.
    """

    def save_fig(name, dpi=300):
        path = os.path.join(output_dir, name)
        plt.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight', format='pdf')
        plt.savefig(path + ".png", dpi=dpi, bbox_inches='tight', format='png')
        print(f"    ✓ Saved: {name}.pdf/.png")
        plt.close()

    print(f"\n" + "═" * 80)
    print(f"  PROCESSING DRUG: {drug_name}")
    print("═" * 80)

    start_time = time.time()

    # Locate the drug in the reference dataset
    print(f"\n  STEP 1: FIND {drug_name} IN REFERENCE DATA")
    print("─" * 50)

    ref_drug_names = list(ref_data.index)
    has_reference = False
    reference_drug_name = None

    if drug_name in ref_drug_names:
        reference_drug_name = drug_name
        has_reference = True
        print(f"  ✅ Exact match: {reference_drug_name}")
    elif drug_name.lower() in [drug.lower() for drug in ref_drug_names]:
        for drug in ref_drug_names:
            if drug.lower() == drug_name.lower():
                reference_drug_name = drug
                has_reference = True
                print(f"  ✅ Case-insensitive match: {reference_drug_name}")
                break
    else:
        print(f"  ❌ No exact match found: {drug_name}")
        has_reference = False

    ref_cluster_target = None
    ref_family_target = None

    if has_reference:
        ref_target_info = ref_info[ref_info['Drug'] == reference_drug_name]
        if not ref_target_info.empty:
            ref_target_row = ref_target_info.iloc[0]
            ref_umap1_target = ref_target_row['UMAP1']
            ref_umap2_target = ref_target_row['UMAP2']
            ref_cluster_target = int(ref_target_row['Cluster'])
            ref_family_target = get_drp_family(ref_cluster_target)
            print(f"  📍 Reference position: UMAP({ref_umap1_target:.3f}, {ref_umap2_target:.3f})")
            print(f"     DRP-C{ref_cluster_target} | Family {ref_family_target}")
        else:
            has_reference = False
            print(f"  ⚠️ Reference drug not found in UMAP coordinates")

    # Load drug profile files
    print(f"\n  STEP 2: LOADING DRUG DATA FROM {drug_name}")
    print("─" * 50)

    csv_files = glob.glob(os.path.join(drug_folder, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith("_extraction_info.csv")]
    csv_files = sorted(csv_files)

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {drug_folder}")

    print(f"  Found {len(csv_files)} cell-line data files")

    cellline_data = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        cellline_name = filename.replace('.csv', '')
        data = pd.read_csv(csv_file, index_col=0)
        cellline_data[cellline_name] = data.T
        print(f"    ✓ {cellline_name}: {data.T.shape}")

    # Protein alignment
    print(f"\n  STEP 3: PROTEIN ALIGNMENT")
    print("─" * 50)

    ref_proteins = set(ref_data.columns)
    aligned_data = {}

    for cellline_name, data in cellline_data.items():
        cellline_proteins = set(data.columns)
        common_proteins = sorted(ref_proteins & cellline_proteins)
        coverage = len(common_proteins) / len(ref_proteins) * 100

        print(f"  {cellline_name}: {len(common_proteins)} shared proteins ({coverage:.1f}%)")

        if len(common_proteins) < 500:
            print(f"    ⚠️ WARNING: Insufficient number of shared proteins ({len(common_proteins)})")
            continue

        ref_aligned = ref_data[common_proteins]
        cellline_aligned = data[common_proteins]

        common_drugs = sorted(set(ref_aligned.index) & set(ref_info['Drug']))
        ref_aligned = ref_aligned.loc[common_drugs]
        ref_info_aligned = ref_info[ref_info['Drug'].isin(common_drugs)].reset_index(drop=True)

        scaler = StandardScaler()
        ref_std = scaler.fit_transform(ref_aligned.values)
        cellline_std = scaler.transform(cellline_aligned.values)

        aligned_data[cellline_name] = {
            'ref_data': ref_aligned,
            'ref_std': ref_std,
            'cellline_data': cellline_aligned,
            'cellline_std': cellline_std,
            'common_proteins': common_proteins,
            'scaler': scaler,
            'ref_info': ref_info_aligned,
            'ref_drug_names': list(ref_aligned.index),
            'cellline_conditions': list(cellline_aligned.index)
        }

    if len(aligned_data) == 0:
        raise ValueError("No valid cell line data after alignment")

    print(f"  Successfully aligned {len(aligned_data)} cell lines")

    # k-NN projection analysis
    print(f"\n  STEP 4: k-NN PROJECTION")
    print("─" * 50)

    all_results = []
    projection_coords = {}

    for cellline_name, aligned in aligned_data.items():
        print(f"    Processing: {cellline_name}")

        cos_sim = cosine_similarity(aligned['cellline_data'].values, aligned['ref_data'].values)
        ref_labels = aligned['ref_info'].set_index('Drug').loc[aligned['ref_drug_names'], 'Cluster'].values
        ref_umap1 = aligned['ref_info'].set_index('Drug').loc[aligned['ref_drug_names'], 'UMAP1'].values
        ref_umap2 = aligned['ref_info'].set_index('Drug').loc[aligned['ref_drug_names'], 'UMAP2'].values

        cellline_results = []
        cellline_coords = []

        for i, condition in enumerate(aligned['cellline_conditions']):
            sims = cos_sim[i]
            topk_idx = np.argsort(sims)[::-1][:K_NN]

            k_drugs = [aligned['ref_drug_names'][j] for j in topk_idx]
            k_sims = sims[topk_idx]
            k_clusters = ref_labels[topk_idx]
            k_u1 = ref_umap1[topk_idx]
            k_u2 = ref_umap2[topk_idx]

            w = np.maximum(k_sims, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(K_NN) / K_NN

            cluster_scores = {}
            for cl, wi in zip(k_clusters, w):
                cluster_scores[cl] = cluster_scores.get(cl, 0) + wi

            best_cl = max(cluster_scores, key=cluster_scores.get)
            confidence = cluster_scores[best_cl]

            proj_u1 = np.sum(w * k_u1)
            proj_u2 = np.sum(w * k_u2)

            projected_family = get_drp_family(best_cl)

            result_row = {
                'Cellline': cellline_name,
                'Condition': condition,
                'Assigned_DRP': int(best_cl),
                'Assigned_Family': projected_family,
                'DRP_Name': DRP_NAMES.get(int(best_cl), "Unknown"),
                'DRP_Family': DRP_FAMILY_FULL_NAMES.get(projected_family, "Unknown"),
                'Confidence': round(confidence, 4),
                'UMAP1_knn': round(proj_u1, 4),
                'UMAP2_knn': round(proj_u2, 4),
                'Top1_Drug': k_drugs[0],
                'Top1_Cosine': round(k_sims[0], 4),
                'Common_Proteins': len(aligned['common_proteins'])
            }

            if has_reference and ref_cluster_target is not None:
                result_row['Reference_DRP'] = ref_cluster_target
                result_row['Reference_Family'] = ref_family_target
                result_row['DRP_Match'] = (int(best_cl) == ref_cluster_target)
                result_row['Family_Match'] = (projected_family == ref_family_target)

                spatial_dist = np.sqrt((proj_u1 - ref_umap1_target) ** 2 +
                                       (proj_u2 - ref_umap2_target) ** 2)
                result_row['Distance_to_Reference'] = round(spatial_dist, 6)

            cellline_results.append(result_row)
            cellline_coords.append((proj_u1, proj_u2, best_cl, condition))

        all_results.extend(cellline_results)
        projection_coords[cellline_name] = cellline_coords

    # Save results
    print(f"\n  STEP 5: SAVING RESULT FILES")
    print("─" * 50)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "projection_results.csv"), index=False)

    processing_time = time.time() - start_time

    summary = {
        'Drug': drug_name,
        'Processing_Time_Seconds': round(processing_time, 2),
        'Cell_Lines_Processed': len(aligned_data),
        'Total_Conditions': len(results_df),
        'Has_Reference_in_HepG2': has_reference,
        'Reference_Drug_Name': reference_drug_name if has_reference else 'N/A',
        'Average_Protein_Coverage': round(
            np.mean([len(data['common_proteins']) for data in aligned_data.values()]) / len(ref_proteins) * 100, 2)
    }

    if has_reference and len(results_df) > 0 and ref_cluster_target is not None:
        if 'DRP_Match' in results_df.columns and 'Family_Match' in results_df.columns:
            drp_match_rate = results_df['DRP_Match'].mean() * 100
            family_match_rate = results_df['Family_Match'].mean() * 100
            avg_distance = results_df['Distance_to_Reference'].mean()

            summary.update({
                'Reference_DRP': ref_cluster_target,
                'Reference_Family': ref_family_target,
                'DRP_Exact_Match_Rate_Percent': round(drp_match_rate, 1),
                'Family_Match_Rate_Percent': round(family_match_rate, 1),
                'Average_Distance_to_Reference': round(avg_distance, 4),
                'DRP_Match_Count': int(results_df['DRP_Match'].sum()),
                'Family_Match_Count': int(results_df['Family_Match'].sum())
            })

    print(f"  ✅ {drug_name} processing completed ({processing_time:.1f}s)")

    return summary


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  In-Depth Analysis Visualization                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def create_deep_analysis_visualization(batch_output_root, all_results_df):
    """
    Create comprehensive visualizations for the in-depth analysis.
    """
    print(f"\n🔬 Generating in-depth analysis visualization...")

    ref_results = all_results_df[
        (all_results_df['DRP_Match'].notna()) &
        (all_results_df['Family_Match'].notna())
    ].copy()

    if len(ref_results) == 0:
        print("  ⚠️ No reference-associated data available for analysis")
        return

    baselines = calculate_random_baselines()

    obs_drp_acc = ref_results['DRP_Match'].mean()
    obs_family_acc = ref_results['Family_Match'].mean()

    drp_test = perform_statistical_test(
        obs_drp_acc,
        baselines['drp_baseline'],
        len(ref_results)
    )

    family_test = perform_statistical_test(
        obs_family_acc,
        baselines['family_baseline'],
        len(ref_results)
    )

    print(f"\n📊 Statistical test results:")
    print(f"  DRP accuracy: {obs_drp_acc:.2%} vs random baseline {baselines['drp_baseline']:.2%}")
    print(f"    - Fold improvement: {drp_test['fold_improvement']:.2f}x")
    print(f"    - P-value: {drp_test['p_value']:.2e} "
          f"{'✅ Significant' if drp_test['significant'] else '❌ Not significant'}")

    print(f"  Family accuracy: {obs_family_acc:.2%} vs random baseline {baselines['family_baseline']:.2%}")
    print(f"    - Fold improvement: {family_test['fold_improvement']:.2f}x")
    print(f"    - P-value: {family_test['p_value']:.2e} "
          f"{'✅ Significant' if family_test['significant'] else '❌ Not significant'}")

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # ========== Row 1: Baseline comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel 1: DRP accuracy vs random baseline
    categories = ['Random\nBaseline', 'Observed\nAccuracy']
    values = [baselines['drp_baseline'], obs_drp_acc]
    colors = ['lightgray', '#E74C3C' if drp_test['significant'] else 'orange']

    bars = ax1.bar(
        categories,
        values,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=2
    )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'DRP Exact Match vs Random Baseline\n'
        f'({drp_test["fold_improvement"]:.1f}x improvement, p={drp_test["p_value"]:.2e})',
        fontsize=11,
        fontweight='bold'
    )

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02,
            f'{val:.1%}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Panel 2: Family accuracy vs random baseline
    values_family = [baselines['family_baseline'], obs_family_acc]
    colors_family = ['lightgray', '#3498DB' if family_test['significant'] else 'orange']

    bars = ax2.bar(
        categories,
        values_family,
        color=colors_family,
        alpha=0.7,
        edgecolor='black',
        linewidth=2
    )

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Family Match vs Random Baseline\n'
        f'({family_test["fold_improvement"]:.1f}x improvement, p={family_test["p_value"]:.2e})',
        fontsize=11,
        fontweight='bold'
    )

    for bar, val in zip(bars, values_family):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02,
            f'{val:.1%}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Panel 3: Confidence distributions for correct vs incorrect predictions
    correct_conf = ref_results[ref_results['DRP_Match'] == True]['Confidence']
    wrong_conf = ref_results[ref_results['DRP_Match'] == False]['Confidence']

    ax3.hist(
        correct_conf,
        bins=30,
        alpha=0.6,
        color='green',
        label=f'Correct (n={len(correct_conf)})',
        edgecolor='black'
    )

    ax3.hist(
        wrong_conf,
        bins=30,
        alpha=0.6,
        color='red',
        label=f'Wrong (n={len(wrong_conf)})',
        edgecolor='black'
    )

    ax3.axvline(
        correct_conf.mean(),
        color='darkgreen',
        linestyle='--',
        linewidth=2,
        label=f'Correct Mean={correct_conf.mean():.3f}'
    )

    ax3.axvline(
        wrong_conf.mean(),
        color='darkred',
        linestyle='--',
        linewidth=2,
        label=f'Wrong Mean={wrong_conf.mean():.3f}'
    )

    ax3.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title(
        'Confidence Distribution: Correct vs Wrong Predictions',
        fontsize=11,
        fontweight='bold'
    )
    ax3.legend()

    # ========== Row 2: Transferability analysis ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Panel 4: Transferability by reference DRP
    transferability_df = analyze_transferability_by_drp(ref_results)

    if transferability_df is not None and len(transferability_df) > 0:
        sorted_trans = transferability_df.sort_values('DRP_Accuracy', ascending=True)

        y_pos = np.arange(len(sorted_trans))
        colors_bar = [DRP_COLORS[drp] for drp in sorted_trans['Reference_DRP']]

        bars = ax4.barh(
            y_pos,
            sorted_trans['DRP_Accuracy'],
            color=colors_bar,
            alpha=0.7,
            edgecolor='black'
        )

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(
            [
                f"C{drp} (n={n})"
                for drp, n in zip(
                    sorted_trans['Reference_DRP'],
                    sorted_trans['N_Samples']
                )
            ],
            fontsize=9
        )

        ax4.set_xlabel('DRP Accuracy', fontsize=11, fontweight='bold')
        ax4.set_title(
            'Transferability by Reference DRP\n(Which DRPs transfer well?)',
            fontsize=11,
            fontweight='bold'
        )
        ax4.axvline(
            0.5,
            color='red',
            linestyle='--',
            alpha=0.5,
            label='50% threshold'
        )
        ax4.set_xlim(0, 1)
        ax4.legend()

        for i, (bar, acc) in enumerate(zip(bars, sorted_trans['DRP_Accuracy'])):
            ax4.text(
                acc + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%}',
                va='center',
                fontsize=9,
                fontweight='bold'
            )

    # Panel 5: Transferability by family
    family_transfer = ref_results.groupby('Reference_Family').agg({
        'DRP_Match': 'mean',
        'Family_Match': 'mean',
        'Confidence': 'mean'
    }).reset_index()

    x_pos = np.arange(len(family_transfer))
    width = 0.35

    bars1 = ax5.bar(
        x_pos - width / 2,
        family_transfer['DRP_Match'],
        width,
        label='DRP Exact',
        color='#E74C3C',
        alpha=0.7,
        edgecolor='black'
    )

    bars2 = ax5.bar(
        x_pos + width / 2,
        family_transfer['Family_Match'],
        width,
        label='Family Match',
        color='#3498DB',
        alpha=0.7,
        edgecolor='black'
    )

    ax5.set_xlabel('Reference Family', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax5.set_title('Transferability by Family', fontsize=11, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(family_transfer['Reference_Family'])
    ax5.legend()
    ax5.set_ylim(0, 1)
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel 6: Failure rate by cell line
    failure_patterns = analyze_failure_patterns(ref_results)

    if failure_patterns and 'by_cellline' in failure_patterns:
        cellline_data = pd.DataFrame(failure_patterns['by_cellline']).T
        cellline_data = cellline_data.sort_values('failure_rate', ascending=False)

        y_pos = np.arange(len(cellline_data))

        bars = ax6.barh(
            y_pos,
            cellline_data['failure_rate'],
            alpha=0.7,
            color='coral',
            edgecolor='black'
        )

        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(
            [
                f"{idx}\n(n={int(row['n_samples'])})"
                for idx, row in cellline_data.iterrows()
            ],
            fontsize=9
        )

        ax6.set_xlabel('Failure Rate', fontsize=11, fontweight='bold')
        ax6.set_title('Failure Rate by Cell Line', fontsize=11, fontweight='bold')
        ax6.set_xlim(0, 1)

        for i, (bar, rate) in enumerate(zip(bars, cellline_data['failure_rate'])):
            ax6.text(
                rate + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{rate:.1%}',
                va='center',
                fontsize=9,
                fontweight='bold'
            )

    # ========== Row 3: Confidence calibration ==========
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    # Panel 7: Confidence calibration curve
    calibration_results = analyze_confidence_calibration(ref_results)

    if calibration_results:
        calib_df = calibration_results['calibration_df']

        ax7.plot(
            [0, 1],
            [0, 1],
            'k--',
            label='Perfect Calibration',
            linewidth=2
        )

        ax7.scatter(
            calib_df['avg_confidence'],
            calib_df['avg_accuracy'],
            s=calib_df['n_samples'] * 2,
            alpha=0.6,
            c='purple',
            edgecolors='black',
            linewidth=1.5
        )

        for _, row in calib_df.iterrows():
            ax7.annotate(
                f"{int(row['n_samples'])}",
                (row['avg_confidence'], row['avg_accuracy']),
                fontsize=8,
                ha='center'
            )

        ax7.set_xlabel('Average Confidence', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
        ax7.set_title(
            f'Confidence Calibration Curve\nECE = {calibration_results["ece"]:.3f}',
            fontsize=11,
            fontweight='bold'
        )
        ax7.legend()
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3)

    # Panel 8: Confidence threshold recommendation
    thresholds_to_test = [0.6, 0.7, 0.8, 0.9]
    threshold_results = []

    for target_prec in thresholds_to_test:
        result = recommend_confidence_threshold(ref_results, target_prec)
        if result and result['achievable']:
            threshold_results.append(result)

    if threshold_results:
        thresh_df = pd.DataFrame(threshold_results)

        ax8_twin = ax8.twinx()
        x_pos = np.arange(len(thresh_df))

        bars = ax8.bar(
            x_pos,
            thresh_df['retention_rate'],
            alpha=0.6,
            color='skyblue',
            label='Retention Rate',
            edgecolor='black'
        )

        line = ax8_twin.plot(
            x_pos,
            thresh_df['actual_precision'],
            'ro-',
            linewidth=2,
            markersize=8,
            label='Actual Precision'
        )

        ax8.set_xlabel('Target Precision Level', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Retention Rate', fontsize=11, fontweight='bold', color='blue')
        ax8_twin.set_ylabel('Actual Precision', fontsize=11, fontweight='bold', color='red')
        ax8.set_title(
            'Confidence Threshold vs Retention\n(How many predictions to keep?)',
            fontsize=11,
            fontweight='bold'
        )

        ax8.set_xticks(x_pos)
        ax8.set_xticklabels([f"{t:.0%}" for t in thresh_df['target_precision']])
        ax8.set_ylim(0, 1)
        ax8_twin.set_ylim(0, 1)

        for i, row in thresh_df.iterrows():
            ax8.text(
                i,
                row['retention_rate'] + 0.02,
                f"θ={row['recommended_threshold']:.2f}",
                ha='center',
                fontsize=8,
                fontweight='bold'
            )

    # Panel 9: Failure rate by protein coverage
    if failure_patterns and 'by_protein_coverage' in failure_patterns:
        coverage_data = pd.DataFrame(failure_patterns['by_protein_coverage']).T
        coverage_data = coverage_data.sort_index()

        if len(coverage_data) > 0:
            bars = ax9.bar(
                range(len(coverage_data)),
                coverage_data['failure_rate'],
                alpha=0.7,
                color='orange',
                edgecolor='black'
            )

            ax9.set_xticks(range(len(coverage_data)))
            ax9.set_xticklabels([
                f"{idx}\n(n={int(row['n_samples'])})"
                for idx, row in coverage_data.iterrows()
            ])

            ax9.set_xlabel('Protein Coverage Bins', fontsize=11, fontweight='bold')
            ax9.set_ylabel('Failure Rate', fontsize=11, fontweight='bold')
            ax9.set_title(
                'Failure Rate by Protein Coverage\n(Does coverage matter?)',
                fontsize=11,
                fontweight='bold'
            )
            ax9.set_ylim(0, 1)

            for bar, rate in zip(bars, coverage_data['failure_rate']):
                height = bar.get_height()
                ax9.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{rate:.1%}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

    # ========== Row 4: Confusion matrices ==========
    ax10 = fig.add_subplot(gs[3, 0])
    ax11 = fig.add_subplot(gs[3, 1])
    ax12 = fig.add_subplot(gs[3, 2])

    # Panel 10: DRP confusion matrix
    drp_cm = confusion_matrix(
        ref_results['Reference_DRP'],
        ref_results['Assigned_DRP'],
        labels=sorted(set(DRP_FAMILY_DICT.keys()))
    )

    im = ax10.imshow(drp_cm, cmap='Blues', aspect='auto')

    ax10.set_xticks(range(len(drp_cm)))
    ax10.set_yticks(range(len(drp_cm)))

    ax10.set_xticklabels(
        [f'C{i}' for i in sorted(set(DRP_FAMILY_DICT.keys()))],
        fontsize=9
    )
    ax10.set_yticklabels(
        [f'C{i}' for i in sorted(set(DRP_FAMILY_DICT.keys()))],
        fontsize=9
    )

    ax10.set_xlabel('Predicted DRP', fontsize=11, fontweight='bold')
    ax10.set_ylabel('Reference DRP', fontsize=11, fontweight='bold')
    ax10.set_title('DRP Confusion Matrix', fontsize=11, fontweight='bold')

    for i in range(len(drp_cm)):
        for j in range(len(drp_cm)):
            text = ax10.text(
                j,
                i,
                drp_cm[i, j],
                ha="center",
                va="center",
                color="white" if drp_cm[i, j] > drp_cm.max() / 2 else "black",
                fontsize=8,
                fontweight='bold'
            )

    plt.colorbar(im, ax=ax10)

    # Panel 11: Family confusion matrix
    family_cm = confusion_matrix(
        ref_results['Reference_Family'],
        ref_results['Assigned_Family'],
        labels=sorted(set(DRP_FAMILY_DICT.values()))
    )

    im = ax11.imshow(family_cm, cmap='Greens', aspect='auto')

    ax11.set_xticks(range(len(family_cm)))
    ax11.set_yticks(range(len(family_cm)))

    family_labels = sorted(set(DRP_FAMILY_DICT.values()))
    ax11.set_xticklabels(family_labels, fontsize=10)
    ax11.set_yticklabels(family_labels, fontsize=10)

    ax11.set_xlabel('Predicted Family', fontsize=11, fontweight='bold')
    ax11.set_ylabel('Reference Family', fontsize=11, fontweight='bold')
    ax11.set_title('Family Confusion Matrix', fontsize=11, fontweight='bold')

    for i in range(len(family_cm)):
        for j in range(len(family_cm)):
            text = ax11.text(
                j,
                i,
                family_cm[i, j],
                ha="center",
                va="center",
                color="white" if family_cm[i, j] > family_cm.max() / 2 else "black",
                fontsize=10,
                fontweight='bold'
            )

    plt.colorbar(im, ax=ax11)

    # Panel 12: Accuracy vs confidence bins
    if failure_patterns and 'by_confidence' in failure_patterns:
        conf_data = pd.DataFrame(failure_patterns['by_confidence']).T
        conf_data = conf_data.sort_index()

        if len(conf_data) > 0:
            x_pos = range(len(conf_data))

            bars = ax12.bar(
                x_pos,
                conf_data['accuracy'],
                alpha=0.7,
                color='purple',
                edgecolor='black'
            )

            ax12.set_xticks(x_pos)
            ax12.set_xticklabels([
                f"{idx}\n(n={int(row['n_samples'])})"
                for idx, row in conf_data.iterrows()
            ], fontsize=9)

            ax12.set_xlabel('Confidence Bins', fontsize=11, fontweight='bold')
            ax12.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax12.set_title(
                'Accuracy vs Confidence Level\n(Can confidence predict correctness?)',
                fontsize=11,
                fontweight='bold'
            )
            ax12.set_ylim(0, 1)
            ax12.axhline(
                0.5,
                color='red',
                linestyle='--',
                alpha=0.5,
                label='Random'
            )
            ax12.legend()

            for bar, acc in zip(bars, conf_data['accuracy']):
                height = bar.get_height()
                ax12.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{acc:.1%}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

    plt.suptitle(
        'Deep Analysis: Transferability, Calibration & Failure Diagnosis',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Save visualization
    output_path = os.path.join(
        batch_output_root,
        "Deep_Analysis_Visualization"
    )

    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ In-depth analysis visualization saved: Deep_Analysis_Visualization.pdf/.png")

    # Save statistical report
    report_path = os.path.join(
        batch_output_root,
        "Statistical_Report.txt"
    )

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. BASELINE COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(ref_results)}\n\n")

        f.write(f"DRP Exact Match:\n")
        f.write(f"  - Observed: {obs_drp_acc:.2%}\n")
        f.write(f"  - Random baseline: {baselines['drp_baseline']:.2%}\n")
        f.write(f"  - Improvement: {drp_test['fold_improvement']:.2f}x\n")
        f.write(f"  - P-value: {drp_test['p_value']:.2e}\n")
        f.write(f"  - Significant: {'YES ✅' if drp_test['significant'] else 'NO ❌'}\n\n")

        f.write(f"Family Match:\n")
        f.write(f"  - Observed: {obs_family_acc:.2%}\n")
        f.write(f"  - Random baseline: {baselines['family_baseline']:.2%}\n")
        f.write(f"  - Improvement: {family_test['fold_improvement']:.2f}x\n")
        f.write(f"  - P-value: {family_test['p_value']:.2e}\n")
        f.write(f"  - Significant: {'YES ✅' if family_test['significant'] else 'NO ❌'}\n\n")

        f.write("\n2. TRANSFERABILITY BY DRP\n")
        f.write("-" * 80 + "\n")

        if transferability_df is not None:
            f.write(transferability_df.to_string())
            f.write("\n\n")

        f.write("\n3. CONFIDENCE THRESHOLD RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")

        for result in threshold_results:
            f.write(f"Target Precision: {result['target_precision']:.0%}\n")
            f.write(f"  - Recommended threshold: {result['recommended_threshold']:.3f}\n")
            f.write(f"  - Retention rate: {result['retention_rate']:.1%}\n")
            f.write(f"  - Actual precision: {result['actual_precision']:.1%}\n\n")

        f.write("\n4. CALIBRATION ERROR\n")
        f.write("-" * 80 + "\n")

        if calibration_results:
            f.write(f"Expected Calibration Error (ECE): {calibration_results['ece']:.4f}\n")
            f.write("(Lower is better; <0.05 is well-calibrated)\n\n")

        f.write("\n5. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")

        if drp_test['significant']:
            f.write("✅ DRP projection shows significant improvement over random baseline.\n")
        else:
            f.write("⚠️ DRP projection does not significantly outperform random baseline.\n")

        if calibration_results and calibration_results['ece'] < 0.1:
            f.write("✅ Confidence scores are reasonably well-calibrated.\n")
        else:
            f.write("⚠️ Confidence scores may not reliably indicate prediction quality.\n")

        if transferability_df is not None:
            high_transfer = transferability_df[
                transferability_df['DRP_Accuracy'] > 0.6
            ]

            low_transfer = transferability_df[
                transferability_df['DRP_Accuracy'] < 0.3
            ]

            if len(high_transfer) > 0:
                f.write(f"\n✅ Highly transferable DRPs (>60% accuracy):\n")

                for _, row in high_transfer.iterrows():
                    f.write(
                        f"   - C{row['Reference_DRP']}: "
                        f"{row['DRP_Accuracy']:.1%}\n"
                    )

            if len(low_transfer) > 0:
                f.write(f"\n⚠️ Poorly transferable DRPs (<30% accuracy):\n")

                for _, row in low_transfer.iterrows():
                    f.write(
                        f"   - C{row['Reference_DRP']}: "
                        f"{row['DRP_Accuracy']:.1%}\n"
                    )

        if threshold_results:
            best_threshold = threshold_results[0]

            f.write(
                f"\n💡 Suggested confidence threshold for reliable predictions:\n"
            )

            f.write(
                f"   θ ≥ {best_threshold['recommended_threshold']:.3f}\n"
            )

            f.write(
                f"   (Retains {best_threshold['retention_rate']:.1%} "
                f"of predictions at "
                f"{best_threshold['actual_precision']:.1%} precision)\n"
            )

    print(f"  ✅ Statistical report saved: Statistical_Report.txt")


def create_batch_summary_visualization(batch_output_root, summary_df, ref_data):
    """
    Create a simplified summary visualization of the batch-processing results.
    """
    print(f"\n📊 Generating batch-processing summary visualization...")

    try:
        successful_df = summary_df[
            summary_df['Status'] == 'Success'
        ].copy()

        if len(successful_df) == 0:
            print("  ⚠️ No successfully processed drugs; skipping summary visualization")
            return

        # Create a simplified summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Processing success rate
        status_counts = summary_df['Status'].value_counts()

        axes[0, 0].pie(
            status_counts.values,
            labels=status_counts.index,
            autopct='%1.1f%%'
        )

        axes[0, 0].set_title('Processing Success Rate')

        # Processing-time distribution
        axes[0, 1].hist(
            successful_df['Processing_Time_Seconds'],
            bins=20,
            edgecolor='black'
        )

        axes[0, 1].set_xlabel('Processing Time (s)')
        axes[0, 1].set_title('Processing Time Distribution')

        # DRP match-rate distribution
        ref_drugs = successful_df[
            successful_df['Has_Reference_in_HepG2'] == True
        ]

        if len(ref_drugs) > 0 and 'DRP_Exact_Match_Rate_Percent' in ref_drugs.columns:
            axes[1, 0].hist(
                ref_drugs['DRP_Exact_Match_Rate_Percent'],
                bins=15,
                edgecolor='black'
            )

            axes[1, 0].set_xlabel('DRP Match Rate (%)')
            axes[1, 0].set_title('DRP Match Rate Distribution')

        # Family match-rate distribution
        if len(ref_drugs) > 0 and 'Family_Match_Rate_Percent' in ref_drugs.columns:
            axes[1, 1].hist(
                ref_drugs['Family_Match_Rate_Percent'],
                bins=15,
                edgecolor='black'
            )

            axes[1, 1].set_xlabel('Family Match Rate (%)')
            axes[1, 1].set_title('Family Match Rate Distribution')

        plt.tight_layout()

        output_path = os.path.join(
            batch_output_root,
            "Batch_Summary_Visualization"
        )

        plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Batch summary visualization saved")

    except Exception as e:
        print(f"  ⚠️ Error generating summary visualization: {e}")


def main():
    """
    Main batch-processing workflow.
    """
    os.makedirs(BATCH_OUTPUT_ROOT, exist_ok=True)

    drug_folders = [
        d for d in os.listdir(DRUG_PROFILES_ROOT)
        if os.path.isdir(os.path.join(DRUG_PROFILES_ROOT, d))
    ]

    drug_folders = sorted(drug_folders)

    print("╔" + "═" * 80 + "╗")
    print("║" + "  BATCH PROCESSING: DEEP DRP PROJECTION ANALYSIS".center(80) + "║")
    print("╠" + "═" * 80 + "╣")
    print(f"║  Found {len(drug_folders)} drug folders".ljust(81) + "║")
    print("╚" + "═" * 80 + "╝")

    # Load reference data
    print(f"\n📂 Loading reference data...")

    try:
        ref_data = pd.read_csv(REF_MATRIX_PATH, index_col=0)
        umap_df = pd.read_csv(UMAP_COORD_PATH)
        label_df = pd.read_csv(LABEL_PATH)

        id_col_umap = umap_df.columns[0]

        u1_candidates = [
            c for c in umap_df.columns
            if any(k in c.lower() for k in ['umap1', 'umap_1', 'umap 1'])
        ]

        u2_candidates = [
            c for c in umap_df.columns
            if any(k in c.lower() for k in ['umap2', 'umap_2', 'umap 2'])
        ]

        u1_col = u1_candidates[0] if u1_candidates else umap_df.columns[1]
        u2_col = u2_candidates[0] if u2_candidates else umap_df.columns[2]

        id_col_label = label_df.columns[0]

        cl_candidates = [
            c for c in label_df.columns
            if any(k in c.lower() for k in ['cluster', 'label', 'group', 'class'])
        ]

        cl_col = cl_candidates[0] if cl_candidates else label_df.columns[1]

        ref_info = umap_df[
            [id_col_umap, u1_col, u2_col]
        ].merge(
            label_df[[id_col_label, cl_col]],
            left_on=id_col_umap,
            right_on=id_col_label,
            how='inner'
        )

        ref_info = ref_info.rename(columns={
            id_col_umap: 'Drug',
            u1_col: 'UMAP1',
            u2_col: 'UMAP2',
            cl_col: 'Cluster'
        })

        ref_info = ref_info[
            ['Drug', 'UMAP1', 'UMAP2', 'Cluster']
        ]

        ref_info['Cluster'] = ref_info['Cluster'].astype(int)

        print(f"✅ Reference data loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load reference data: {e}")
        return

    # Batch processing
    batch_start_time = time.time()
    processing_summary = []
    failed_drugs = []

    for i, drug_folder_name in enumerate(drug_folders, 1):

        print(f"\n{'=' * 80}")
        print(
            f"  [{i}/{len(drug_folders)}] "
            f"Processing: {drug_folder_name}"
        )
        print(f"{'=' * 80}")

        drug_folder_path = os.path.join(
            DRUG_PROFILES_ROOT,
            drug_folder_name
        )

        drug_output_dir = os.path.join(
            BATCH_OUTPUT_ROOT,
            drug_folder_name
        )

        os.makedirs(drug_output_dir, exist_ok=True)

        try:
            summary = process_single_drug(
                drug_name=drug_folder_name,
                drug_folder=drug_folder_path,
                output_dir=drug_output_dir,
                ref_data=ref_data,
                umap_df=umap_df,
                label_df=label_df,
                ref_info=ref_info
            )

            summary['Status'] = 'Success'
            summary['Error_Message'] = ''
            processing_summary.append(summary)

        except Exception as e:
            error_msg = str(e)

            print(
                f"❌ Failed to process "
                f"{drug_folder_name}: {error_msg}"
            )

            failed_drugs.append({
                'Drug': drug_folder_name,
                'Status': 'Failed',
                'Error_Message': error_msg,
                'Processing_Time_Seconds': 0,
                'Cell_Lines_Processed': 0,
                'Total_Conditions': 0
            })

    # Save batch summary
    all_summaries = processing_summary + failed_drugs
    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv(
        os.path.join(
            BATCH_OUTPUT_ROOT,
            "Batch_Processing_Summary.csv"
        ),
        index=False
    )

    # Generate batch summary visualization
    create_batch_summary_visualization(
        BATCH_OUTPUT_ROOT,
        summary_df,
        ref_data
    )

    # Combine all projection results for in-depth analysis
    print(f"\n🔬 Combining all projection results for in-depth analysis...")

    all_results = []

    for drug_folder_name in drug_folders:

        result_file = os.path.join(
            BATCH_OUTPUT_ROOT,
            drug_folder_name,
            "projection_results.csv"
        )

        if os.path.exists(result_file):
            try:
                df = pd.read_csv(result_file)
                df['Drug_Folder'] = drug_folder_name
                all_results.append(df)

            except Exception as e:
                print(
                    f"  ⚠️ Failed to read results for "
                    f"{drug_folder_name}: {e}"
                )

    if all_results:
        all_results_df = pd.concat(
            all_results,
            ignore_index=True
        )

        all_results_df.to_csv(
            os.path.join(
                BATCH_OUTPUT_ROOT,
                "All_Projection_Results.csv"
            ),
            index=False
        )

        # Generate in-depth analysis visualization
        create_deep_analysis_visualization(
            BATCH_OUTPUT_ROOT,
            all_results_df
        )

    else:
        print("  ⚠️ No result files found")

    total_time = time.time() - batch_start_time

    # Final report
    print(f"\n")

    print("╔" + "═" * 80 + "╗")
    print("║" + "  BATCH PROCESSING COMPLETED".center(80) + "║")
    print("╠" + "═" * 80 + "╣")

    print(
        f"║  Total runtime: "
        f"{total_time / 3600:.1f} hours".ljust(81) + "║"
    )

    print(
        f"║  Successful: {len(processing_summary)} | "
        f"Failed: {len(failed_drugs)}".ljust(81) + "║"
    )

    print(f"║  Output files:".ljust(81) + "║")

    print(
        f"║    - Batch_Processing_Summary.csv".ljust(81)
        + "║"
    )

    print(
        f"║    - All_Projection_Results.csv".ljust(81)
        + "║"
    )

    print(
        f"║    - Deep_Analysis_Visualization.pdf/.png".ljust(81)
        + "║"
    )

    print(
        f"║    - Statistical_Report.txt".ljust(81)
        + "║"
    )

    print("╚" + "═" * 80 + "╝")


if __name__ == "__main__":
    main()
