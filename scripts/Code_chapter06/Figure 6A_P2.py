"""
═══════════════════════════════════════════════════════════════════════════════
  Batch Processing: Multi-Level Zero-Leakage Benchmark + Three-Tier Statistical
  Tests + Multi-Dimensional Metrics + Stratified Bootstrap
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report
)

from scipy.stats import (
    pearsonr, chi2_contingency, ttest_rel, wilcoxon,
    mannwhitneyu, norm
)

try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as binomtest

from collections import Counter, defaultdict
import os
import glob
import time
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration Section                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter06"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter06"
REF_MATRIX_PATH = DATA_ROOT / "01_HepG2_367_Drugs_log2FC_6682_for_UMAP.csv"
UMAP_COORD_PATH = DATA_ROOT / "HepG2_coordinate system.csv"
LABEL_PATH = DATA_ROOT / "HepG2_label.csv"

DRUG_PROFILES_ROOT = DATA_ROOT / "drug_profiles"
BATCH_OUTPUT_ROOT = RESULTS_ROOT / "02_drug_profiles_leave one group out"

COMPOUND_INFO_PATH = DATA_ROOT / "Table S1-Compound information.csv"
TANIMOTO_MATRIX_PATH = DATA_ROOT / "tanimoto_similarity_matrix.csv"

K_NN = 3
TANIMOTO_THRESHOLD = 0.85
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000

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

DRP_COLORS = {
    1: "#E41A1C", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#FF7F00",
    6: "#A65628", 7: "#F781BF", 8: "#808080", 9: "#66C2A5", 10: "#E6AB02",
}

def get_drp_family(drp_cluster):
    return DRP_FAMILY_DICT.get(int(drp_cluster), "Unknown")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Multi-Dimensional Evaluation Metrics                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics"""
    accuracy = np.mean(y_true == y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

    return {
        'accuracy': accuracy * 100,
        'balanced_accuracy': balanced_acc * 100,
        'macro_f1': macro_f1 * 100,
        'cohen_kappa': kappa,
        'confusion_matrix': cm,
        'n_samples': len(y_true),
        'n_classes_true': len(unique_true),
        'n_classes_pred': len(unique_pred),
        'class_distribution_true': dict(zip(unique_true, counts_true)),
        'class_distribution_pred': dict(zip(unique_pred, counts_pred))
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Label Permutation Empirical Baseline Test                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def permutation_test_baseline(y_true, y_pred, n_permutations=10000,
                              metrics=['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']):
    """Generate empirical null distribution via label permutation"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    observed_metrics = calculate_comprehensive_metrics(y_true, y_pred)
    permutation_results = {metric: [] for metric in metrics}

    for i in range(n_permutations):
        y_true_permuted = np.random.permutation(y_true)
        perm_metrics = calculate_comprehensive_metrics(y_true_permuted, y_pred)

        for metric in metrics:
            permutation_results[metric].append(perm_metrics[metric])

    results = {}
    for metric in metrics:
        perm_dist = np.array(permutation_results[metric])
        observed_value = observed_metrics[metric]
        n_extreme = np.sum(perm_dist >= observed_value)
        p_value = (n_extreme + 1) / (n_permutations + 1)

        empirical_mean = np.mean(perm_dist)
        empirical_std = np.std(perm_dist)
        empirical_95ci = np.percentile(perm_dist, [2.5, 97.5])

        if empirical_std > 0:
            standardized_effect = (observed_value - empirical_mean) / empirical_std
        else:
            standardized_effect = 0

        results[metric] = {
            'observed': observed_value,
            'empirical_mean': empirical_mean,
            'empirical_std': empirical_std,
            'empirical_95ci_lower': empirical_95ci[0],
            'empirical_95ci_upper': empirical_95ci[1],
            'p_value': p_value,
            'significant': p_value < 0.05,
            'standardized_effect': standardized_effect,
            'fold_improvement': observed_value / empirical_mean if empirical_mean > 0 else np.inf,
            'permutation_distribution': perm_dist
        }

    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ⭐ NEW: Stratified Bootstrap Confidence Intervals                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def stratified_bootstrap_confidence_interval(y_true, y_pred, compound_ids,
                                            n_bootstrap=10000, ci=95,
                                            metrics=['accuracy', 'balanced_accuracy',
                                                   'macro_f1', 'cohen_kappa']):
    """
    ⭐ Stratified Bootstrap: Resample compounds within each reference DRP.

    Parameters:
    -----------
    y_true : array-like
        True labels (DRP or Family)
    y_pred : array-like
        Predicted labels
    compound_ids : array-like
        Compound ID corresponding to each profile (for block resampling)
    n_bootstrap : int
        Number of Bootstrap iterations
    ci : int
        Confidence level
    metrics : list
        Metrics to calculate

    Returns:
    --------
    dict : Contains bootstrap CIs for each metric

    Note:
    -----
    This method independently resamples compounds within each reference DRP
    to ensure that every bootstrap sample contains all DRP classes, enabling
    stable computation of balanced accuracy and macro-F1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    compound_ids = np.array(compound_ids)

    # ⭐ Key: Group compounds by reference DRP
    unique_drps = np.unique(y_true)
    drp_to_compounds = defaultdict(set)

    for yt, cid in zip(y_true, compound_ids):
        drp_to_compounds[yt].add(cid)

    # Check if there are enough compounds in each DRP
    min_compounds_per_drp = min(len(compounds) for compounds in drp_to_compounds.values())
    if min_compounds_per_drp < 1:
        print(f"  ⚠️  WARNING: Some DRPs only have {min_compounds_per_drp} compounds; bootstrap may be unstable.")

    # Store bootstrap results
    bootstrap_results = {metric: [] for metric in metrics}

    print(f"    Executing Stratified Bootstrap (n={n_bootstrap})...")
    print(f"    DRP Distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    for iteration in range(n_bootstrap):
        if (iteration + 1) % 2000 == 0:
            print(f"      Progress: {iteration + 1}/{n_bootstrap}")

        # ⭐ Resample compounds within each DRP
        resampled_indices = []

        for drp in unique_drps:
            # All compounds in this DRP
            compounds_in_drp = list(drp_to_compounds[drp])
            n_compounds = len(compounds_in_drp)

            # Resampling with replacement
            resampled_compounds = np.random.choice(
                compounds_in_drp,
                size=n_compounds,
                replace=True
            )

            # Collect all profiles for these compounds
            for compound in resampled_compounds:
                # Find all profile indices for this compound
                profile_indices = np.where(compound_ids == compound)[0]
                resampled_indices.extend(profile_indices)

        # Calculate metrics using resampled indices
        y_true_boot = y_true[resampled_indices]
        y_pred_boot = y_pred[resampled_indices]

        boot_metrics = calculate_comprehensive_metrics(y_true_boot, y_pred_boot)

        for metric in metrics:
            bootstrap_results[metric].append(boot_metrics[metric])

    # Calculate confidence intervals
    alpha = (100 - ci) / 2
    results = {}

    for metric in metrics:
        boot_dist = np.array(bootstrap_results[metric])

        results[metric] = {
            'mean': np.mean(boot_dist),
            'std': np.std(boot_dist),
            'ci_lower': np.percentile(boot_dist, alpha),
            'ci_upper': np.percentile(boot_dist, 100 - alpha),
            'ci_level': ci,
            'bootstrap_distribution': boot_dist  # Save for visualization
        }

    return results


def bootstrap_confidence_interval(data, n_bootstrap=10000, ci=95):
    """Simple Bootstrap Confidence Interval (for paired differences, etc.)"""
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrapped_means, alpha)
    upper = np.percentile(bootstrapped_means, 100 - alpha)

    return {
        'mean': np.mean(data),
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_level': ci
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Tier 2: Paired Comparisons                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def paired_comparison_comprehensive(data1, data2, strategy1_name, strategy2_name):
    """Comprehensive paired comparison: Parametric + Non-parametric tests"""
    if len(data1) != len(data2) or len(data1) < 3:
        return None

    data1 = np.array(data1)
    data2 = np.array(data2)
    diff = data1 - data2

    try:
        t_stat, t_pvalue = ttest_rel(data1, data2)
    except:
        t_stat, t_pvalue = np.nan, 1.0

    try:
        non_zero_diff = diff[diff != 0]
        if len(non_zero_diff) >= 3:
            w_stat, w_pvalue = wilcoxon(non_zero_diff, alternative='two-sided')
        else:
            w_stat, w_pvalue = np.nan, 1.0
    except:
        w_stat, w_pvalue = np.nan, 1.0

    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    cliffs_delta = compute_cliffs_delta(data1, data2)
    bootstrap_result = bootstrap_confidence_interval(diff, n_bootstrap=10000)

    n_wins = np.sum(diff > 0)
    n_losses = np.sum(diff < 0)
    n_ties = np.sum(diff == 0)

    return {
        'strategy1': strategy1_name,
        'strategy2': strategy2_name,
        'n_pairs': len(data1),
        'mean_diff': np.mean(diff),
        'median_diff': np.median(diff),
        'std_diff': np.std(diff, ddof=1),
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        't_significant': t_pvalue < 0.05 if not np.isnan(t_pvalue) else False,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'wilcoxon_significant': w_pvalue < 0.05 if not np.isnan(w_pvalue) else False,
        'cohens_d': cohens_d,
        'cohens_d_interpretation': interpret_cohens_d(cohens_d),
        'cliffs_delta': cliffs_delta,
        'cliffs_delta_interpretation': interpret_cliffs_delta(cliffs_delta),
        'ci_lower': bootstrap_result['ci_lower'],
        'ci_upper': bootstrap_result['ci_upper'],
        'ci_excludes_zero': not (bootstrap_result['ci_lower'] <= 0 <= bootstrap_result['ci_upper']),
        'n_wins': n_wins,
        'n_losses': n_losses,
        'n_ties': n_ties,
        'win_rate': n_wins / len(data1) * 100
    }


def compute_cliffs_delta(x, y):
    """Compute Cliff's Delta (Non-parametric effect size)"""
    n_x = len(x)
    n_y = len(y)
    dominance = 0

    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1

    delta = dominance / (n_x * n_y)
    return delta


def interpret_cohens_d(d):
    """Interpret Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_cliffs_delta(delta):
    """Interpret Cliff's Delta"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "Negligible"
    elif abs_delta < 0.33:
        return "Small"
    elif abs_delta < 0.474:
        return "Medium"
    else:
        return "Large"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Load Compound Grouping Info                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_compound_grouping_info():
    """Load compound grouping information"""
    grouping_info = {
        'compound_annotation': None,
        'tanimoto_matrix': None,
        'api_groups': {},
        'similarity_clusters': {},
        'moa_groups': {}
    }

    if COMPOUND_INFO_PATH and os.path.exists(COMPOUND_INFO_PATH):
        compound_df = pd.read_csv(COMPOUND_INFO_PATH)
        grouping_info['compound_annotation'] = compound_df

        for _, row in compound_df.iterrows():
            drug_id = row.get('Compound_ID', row.get('Compound_Name', ''))
            drug_name = row.get('Compound_Name', drug_id)

            api = row.get('Active_Ingredient', drug_name)
            if pd.isna(api) or api == '':
                api = drug_name
            grouping_info['api_groups'][drug_name] = api

            target = row.get('Primary_Target', 'Unknown')
            if pd.isna(target) or target == '':
                target = 'Unknown'
            grouping_info['moa_groups'][drug_name] = target

        print(f"✓ Loaded compound annotations: {len(compound_df)} compounds")

    if TANIMOTO_MATRIX_PATH and os.path.exists(TANIMOTO_MATRIX_PATH):
        tanimoto_df = pd.read_csv(TANIMOTO_MATRIX_PATH, index_col=0)
        grouping_info['tanimoto_matrix'] = tanimoto_df

        for drug in tanimoto_df.index:
            similar_drugs = tanimoto_df.loc[drug][tanimoto_df.loc[drug] > TANIMOTO_THRESHOLD].index.tolist()
            grouping_info['similarity_clusters'][drug] = similar_drugs

        print(f"✓ Loaded Tanimoto matrix: {tanimoto_df.shape}")

    return grouping_info


def get_drugs_to_exclude(query_drug_name, grouping_info, strategy='identity'):
    """Get the list of drugs to exclude based on the strategy"""
    if strategy == 'no_exclusion':
        return set()

    excluded_drugs = set([query_drug_name])

    if strategy in ['api', 'combined']:
        if query_drug_name in grouping_info['api_groups']:
            query_api = grouping_info['api_groups'][query_drug_name]
            for drug, api in grouping_info['api_groups'].items():
                if api == query_api:
                    excluded_drugs.add(drug)

    if strategy in ['chemical', 'combined']:
        if query_drug_name in grouping_info['similarity_clusters']:
            excluded_drugs.update(grouping_info['similarity_clusters'][query_drug_name])

    if strategy in ['moa', 'combined']:
        if query_drug_name in grouping_info['moa_groups']:
            query_moa = grouping_info['moa_groups'][query_drug_name]
            if query_moa != 'Unknown':
                for drug, moa in grouping_info['moa_groups'].items():
                    if moa == query_moa:
                        excluded_drugs.add(drug)

    return excluded_drugs


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Process a Single Drug (Save complete predictions & compound IDs)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def process_single_drug_with_exclusion(drug_name, drug_folder, output_dir,
                                       ref_data, umap_df, label_df, ref_info,
                                       grouping_info,
                                       strategies=['no_exclusion', 'identity', 'api', 'chemical', 'moa']):
    """Process a single drug (⭐ Save compound IDs for stratified bootstrap)"""
    print(f"\n{'═' * 80}")
    print(f"  PROCESSING: {drug_name}")
    print(f"{'═' * 80}")

    # Look for reference
    ref_drug_names = list(ref_data.index)
    has_reference = False
    reference_drug_name = None

    if drug_name in ref_drug_names:
        reference_drug_name = drug_name
        has_reference = True
    elif drug_name.lower() in [drug.lower() for drug in ref_drug_names]:
        for drug in ref_drug_names:
            if drug.lower() == drug_name.lower():
                reference_drug_name = drug
                has_reference = True
                break

    ref_cluster_target = None
    ref_family_target = None

    if has_reference:
        ref_target_info = ref_info[ref_info['Drug'] == reference_drug_name]
        if not ref_target_info.empty:
            ref_target_row = ref_target_info.iloc[0]
            ref_cluster_target = int(ref_target_row['Cluster'])
            ref_family_target = get_drp_family(ref_cluster_target)

    # Read and align data
    csv_files = glob.glob(os.path.join(drug_folder, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith("_extraction_info.csv")]

    cellline_data = {}
    for csv_file in csv_files:
        cellline_name = os.path.basename(csv_file).replace('.csv', '')
        data = pd.read_csv(csv_file, index_col=0)
        cellline_data[cellline_name] = data.T

    ref_proteins = set(ref_data.columns)
    aligned_data = {}

    for cellline_name, data in cellline_data.items():
        common_proteins = sorted(ref_proteins & set(data.columns))
        if len(common_proteins) < 500:
            continue

        ref_aligned = ref_data[common_proteins]
        cellline_aligned = data[common_proteins]

        common_drugs = sorted(set(ref_aligned.index) & set(ref_info['Drug']))
        ref_aligned = ref_aligned.loc[common_drugs]
        ref_info_aligned = ref_info[ref_info['Drug'].isin(common_drugs)].reset_index(drop=True)

        scaler = StandardScaler()
        scaler.fit_transform(ref_aligned.values)
        scaler.transform(cellline_aligned.values)

        aligned_data[cellline_name] = {
            'ref_data': ref_aligned,
            'cellline_data': cellline_aligned,
            'ref_info': ref_info_aligned,
            'ref_drug_names': list(ref_aligned.index),
            'cellline_conditions': list(cellline_aligned.index)
        }

    # Multi-strategy projection
    all_strategy_results = {}
    strategy_predictions = {}

    for strategy in strategies:
        excluded_drugs = get_drugs_to_exclude(drug_name, grouping_info, strategy)
        strategy_results = []

        # ⭐ Collect predictions, true labels, and compound IDs
        y_true_drp = []
        y_pred_drp = []
        y_true_family = []
        y_pred_family = []
        compound_ids_drp = []  # ⭐ NEW: Record which compound each profile belongs to
        compound_ids_family = []

        for cellline_name, aligned in aligned_data.items():
            available_ref_drugs = [d for d in aligned['ref_drug_names'] if d not in excluded_drugs]

            if len(available_ref_drugs) < K_NN:
                continue

            ref_mask = [d in available_ref_drugs for d in aligned['ref_drug_names']]
            ref_data_filtered = aligned['ref_data'].iloc[ref_mask]
            ref_info_filtered = aligned['ref_info'][aligned['ref_info']['Drug'].isin(available_ref_drugs)]

            cos_sim = cosine_similarity(aligned['cellline_data'].values, ref_data_filtered.values)
            ref_labels = ref_info_filtered.set_index('Drug').loc[available_ref_drugs, 'Cluster'].values

            for i, condition in enumerate(aligned['cellline_conditions']):
                sims = cos_sim[i]
                topk_idx = np.argsort(sims)[::-1][:K_NN]

                k_clusters = ref_labels[topk_idx]
                k_sims = sims[topk_idx]

                w = np.maximum(k_sims, 0)
                w = w / w.sum() if w.sum() > 0 else np.ones(K_NN) / K_NN

                cluster_scores = {}
                for cl, wi in zip(k_clusters, w):
                    cluster_scores[cl] = cluster_scores.get(cl, 0) + wi
                best_cl = max(cluster_scores, key=cluster_scores.get)

                projected_family = get_drp_family(best_cl)

                # ⭐ Generate a unique compound ID (drug_name + cellline + condition)
                compound_id = f"{drug_name}_{cellline_name}_{condition}"

                result_row = {
                    'Strategy': strategy,
                    'Cellline': cellline_name,
                    'Condition': condition,
                    'Assigned_DRP': int(best_cl),
                    'Assigned_Family': projected_family,
                    'Compound_ID': compound_id  # ⭐ Save ID
                }

                if has_reference and ref_cluster_target is not None:
                    result_row['Reference_DRP'] = ref_cluster_target
                    result_row['Reference_Family'] = ref_family_target
                    result_row['DRP_Match'] = (int(best_cl) == ref_cluster_target)
                    result_row['Family_Match'] = (projected_family == ref_family_target)

                    # ⭐ Collect labels and IDs
                    y_true_drp.append(ref_cluster_target)
                    y_pred_drp.append(int(best_cl))
                    compound_ids_drp.append(compound_id)

                    y_true_family.append(ref_family_target)
                    y_pred_family.append(projected_family)
                    compound_ids_family.append(compound_id)

                strategy_results.append(result_row)

        all_strategy_results[strategy] = strategy_results

        # ⭐ Save prediction results (including compound IDs)
        if has_reference and len(y_true_drp) > 0:
            strategy_predictions[strategy] = {
                'y_true_drp': np.array(y_true_drp),
                'y_pred_drp': np.array(y_pred_drp),
                'compound_ids_drp': np.array(compound_ids_drp),
                'y_true_family': np.array(y_true_family),
                'y_pred_family': np.array(y_pred_family),
                'compound_ids_family': np.array(compound_ids_family)
            }

    # Save results
    all_results_combined = []
    for strategy, results in all_strategy_results.items():
        all_results_combined.extend(results)

    if len(all_results_combined) > 0:
        results_df = pd.DataFrame(all_results_combined)
        results_df.to_csv(os.path.join(output_dir, "projection_results_multi_strategy.csv"), index=False)

    # Summarize
    summary = {
        'Drug': drug_name,
        'Has_Reference': has_reference,
        'N_Strategies': len(strategies)
    }

    for strategy in strategies:
        if strategy in strategy_predictions:
            preds = strategy_predictions[strategy]

            drp_metrics = calculate_comprehensive_metrics(
                preds['y_true_drp'],
                preds['y_pred_drp']
            )
            summary[f'{strategy}_DRP_Accuracy'] = drp_metrics['accuracy']
            summary[f'{strategy}_DRP_Balanced_Accuracy'] = drp_metrics['balanced_accuracy']
            summary[f'{strategy}_DRP_Macro_F1'] = drp_metrics['macro_f1']
            summary[f'{strategy}_DRP_Cohen_Kappa'] = drp_metrics['cohen_kappa']

            family_metrics = calculate_comprehensive_metrics(
                preds['y_true_family'],
                preds['y_pred_family']
            )
            summary[f'{strategy}_Family_Accuracy'] = family_metrics['accuracy']
            summary[f'{strategy}_Family_Balanced_Accuracy'] = family_metrics['balanced_accuracy']
            summary[f'{strategy}_Family_Macro_F1'] = family_metrics['macro_f1']
            summary[f'{strategy}_Family_Cohen_Kappa'] = family_metrics['cohen_kappa']

    # ⭐ Save predicted labels and compound IDs
    if has_reference and strategy_predictions:
        predictions_path = os.path.join(output_dir, "predictions_for_confusion_matrix.npz")
        np.savez(predictions_path, **strategy_predictions)

    return summary


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ⭐ ENHANCED: Three-Tier Statistical Tests (Add Stratified Bootstrap CI)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def perform_three_tier_statistical_tests(batch_output_root, summary_df):
    """Three-Tier Statistical Tests (⭐ Adding Stratified Bootstrap CI)"""
    print(f"\n{'═' * 80}")
    print(f"  Three-Tier Statistical Tests (Multi-metrics + Permutation + ⭐ Stratified Boot CI)")
    print(f"{'═' * 80}")

    ref_drugs = summary_df[summary_df['Has_Reference'] == True].copy()

    if len(ref_drugs) == 0:
        print("  ⚠️ No referenced drugs available.")
        return None

    # Detect available strategies
    available_strategies = []
    for strategy in ['no_exclusion', 'identity', 'api', 'chemical', 'moa']:
        if f'{strategy}_DRP_Accuracy' in ref_drugs.columns:
            available_strategies.append(strategy)

    if len(available_strategies) == 0:
        return None

    print(f"\n  Available strategies: {available_strategies}")
    print(f"  Number of referenced drugs: {len(ref_drugs)}")

    # ⭐ Collect prediction results for all drugs (including compound IDs)
    all_predictions = {
        strategy: {
            'y_true_drp': [], 'y_pred_drp': [], 'compound_ids_drp': [],
            'y_true_family': [], 'y_pred_family': [], 'compound_ids_family': []
        }
        for strategy in available_strategies
    }

    for idx, row in ref_drugs.iterrows():
        drug_name = row['Drug']
        drug_output_dir = os.path.join(batch_output_root, drug_name)
        predictions_path = os.path.join(drug_output_dir, "predictions_for_confusion_matrix.npz")

        if os.path.exists(predictions_path):
            data = np.load(predictions_path, allow_pickle=True)
            for strategy in available_strategies:
                if strategy in data.files:
                    strategy_data = data[strategy].item()
                    all_predictions[strategy]['y_true_drp'].extend(strategy_data['y_true_drp'])
                    all_predictions[strategy]['y_pred_drp'].extend(strategy_data['y_pred_drp'])
                    all_predictions[strategy]['compound_ids_drp'].extend(strategy_data['compound_ids_drp'])

                    all_predictions[strategy]['y_true_family'].extend(strategy_data['y_true_family'])
                    all_predictions[strategy]['y_pred_family'].extend(strategy_data['y_pred_family'])
                    all_predictions[strategy]['compound_ids_family'].extend(strategy_data['compound_ids_family'])

    # ============================================================
    # Tier 1: Label Permutation Test + ⭐ Stratified Bootstrap CI
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  Tier 1: Label Permutation Test + ⭐ Stratified Bootstrap CI")
    print(f"  (n_permutations={N_PERMUTATIONS}, n_bootstrap={N_BOOTSTRAP})")
    print(f"{'─' * 80}")

    baseline_tests = []
    permutation_distributions = {}
    bootstrap_distributions = {}  # ⭐ Save bootstrap distributions

    for strategy in available_strategies:
        if len(all_predictions[strategy]['y_true_drp']) == 0:
            continue

        y_true_drp = np.array(all_predictions[strategy]['y_true_drp'])
        y_pred_drp = np.array(all_predictions[strategy]['y_pred_drp'])
        compound_ids_drp = np.array(all_predictions[strategy]['compound_ids_drp'])

        y_true_family = np.array(all_predictions[strategy]['y_true_family'])
        y_pred_family = np.array(all_predictions[strategy]['y_pred_family'])
        compound_ids_family = np.array(all_predictions[strategy]['compound_ids_family'])

        print(f"\n  {strategy.upper()}:")
        print(f"    Sample count: {len(y_true_drp)}")

        # ────────────────────────────────────────────────────
        # DRP Level: Label Permutation
        # ────────────────────────────────────────────────────
        print(f"    DRP Level Label Permutation Test...")
        drp_perm_results = permutation_test_baseline(
            y_true_drp, y_pred_drp,
            n_permutations=N_PERMUTATIONS,
            metrics=['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']
        )

        # ⭐ DRP Level: Stratified Bootstrap CI
        print(f"    DRP Level Stratified Bootstrap CI...")
        drp_boot_results = stratified_bootstrap_confidence_interval(
            y_true_drp, y_pred_drp, compound_ids_drp,
            n_bootstrap=N_BOOTSTRAP,
            ci=95,
            metrics=['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']
        )

        # ────────────────────────────────────────────────────
        # Family Level: Label Permutation
        # ────────────────────────────────────────────────────
        print(f"    Family Level Label Permutation Test...")
        family_perm_results = permutation_test_baseline(
            y_true_family, y_pred_family,
            n_permutations=N_PERMUTATIONS,
            metrics=['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']
        )

        # ⭐ Family Level: Stratified Bootstrap CI
        print(f"    Family Level Stratified Bootstrap CI...")
        family_boot_results = stratified_bootstrap_confidence_interval(
            y_true_family, y_pred_family, compound_ids_family,
            n_bootstrap=N_BOOTSTRAP,
            ci=95,
            metrics=['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']
        )

        # Save distributions
        permutation_distributions[strategy] = {
            'drp': drp_perm_results,
            'family': family_perm_results
        }

        bootstrap_distributions[strategy] = {
            'drp': drp_boot_results,
            'family': family_boot_results
        }

        # ────────────────────────────────────────────────────
        # Summarize results (including Bootstrap CI)
        # ────────────────────────────────────────────────────
        for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']:
            # DRP
            drp_perm = drp_perm_results[metric]
            drp_boot = drp_boot_results[metric]

            baseline_tests.append({
                'Strategy': strategy.upper(),
                'Metric': 'DRP',
                'Indicator': metric,
                'Observed': drp_perm['observed'],
                'Bootstrap_Mean': drp_boot['mean'],  # ⭐ NEW
                'Bootstrap_CI_Lower': drp_boot['ci_lower'],  # ⭐ NEW
                'Bootstrap_CI_Upper': drp_boot['ci_upper'],  # ⭐ NEW
                'Empirical_Mean': drp_perm['empirical_mean'],
                'Empirical_Std': drp_perm['empirical_std'],
                'Empirical_95CI_Lower': drp_perm['empirical_95ci_lower'],
                'Empirical_95CI_Upper': drp_perm['empirical_95ci_upper'],
                'Fold_Improvement': drp_perm['fold_improvement'],
                'Standardized_Effect': drp_perm['standardized_effect'],
                'P_Value': drp_perm['p_value'],
                'Significant': '✅' if drp_perm['significant'] else '❌',
                'N_Samples': len(y_true_drp)
            })

            # Family
            fam_perm = family_perm_results[metric]
            fam_boot = family_boot_results[metric]

            baseline_tests.append({
                'Strategy': strategy.upper(),
                'Metric': 'Family',
                'Indicator': metric,
                'Observed': fam_perm['observed'],
                'Bootstrap_Mean': fam_boot['mean'],  # ⭐ NEW
                'Bootstrap_CI_Lower': fam_boot['ci_lower'],  # ⭐ NEW
                'Bootstrap_CI_Upper': fam_boot['ci_upper'],  # ⭐ NEW
                'Empirical_Mean': fam_perm['empirical_mean'],
                'Empirical_Std': fam_perm['empirical_std'],
                'Empirical_95CI_Lower': fam_perm['empirical_95ci_lower'],
                'Empirical_95CI_Upper': fam_perm['empirical_95ci_upper'],
                'Fold_Improvement': fam_perm['fold_improvement'],
                'Standardized_Effect': fam_perm['standardized_effect'],
                'P_Value': fam_perm['p_value'],
                'Significant': '✅' if fam_perm['significant'] else '❌',
                'N_Samples': len(y_true_family)
            })

        # Print key results (⭐ including Bootstrap CI)
        print(f"    DRP Balanced Accuracy:")
        print(f"      Observed:  {drp_perm_results['balanced_accuracy']['observed']:.2f}%")
        print(f"      Boot CI:   [{drp_boot_results['balanced_accuracy']['ci_lower']:.2f}%, "
              f"{drp_boot_results['balanced_accuracy']['ci_upper']:.2f}%] ⭐")
        print(f"      Null Mean: {drp_perm_results['balanced_accuracy']['empirical_mean']:.2f}%")
        print(f"      Null 95%:  [{drp_perm_results['balanced_accuracy']['empirical_95ci_lower']:.2f}%, "
              f"{drp_perm_results['balanced_accuracy']['empirical_95ci_upper']:.2f}%]")
        print(f"      P-value:   {drp_perm_results['balanced_accuracy']['p_value']:.3e} "
              f"{'✅' if drp_perm_results['balanced_accuracy']['significant'] else '❌'}")

    baseline_df = pd.DataFrame(baseline_tests)

    # ============================================================
    # Tier 2: Pairwise Comparison Between Strategies
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  Tier 2: Pairwise Comparison Between Strategies (Multi-metrics)")
    print(f"{'─' * 80}")

    pairwise_tests = []
    strategy_order = [s for s in ['no_exclusion', 'identity', 'api', 'chemical', 'moa']
                     if s in available_strategies]

    for metric_name in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'Cohen_Kappa']:
        for i in range(len(strategy_order) - 1):
            strategy1 = strategy_order[i]
            strategy2 = strategy_order[i + 1]

            drp_col1 = f'{strategy1}_DRP_{metric_name}'
            drp_col2 = f'{strategy2}_DRP_{metric_name}'

            if drp_col1 not in ref_drugs.columns or drp_col2 not in ref_drugs.columns:
                continue

            valid_drugs = ref_drugs[[drp_col1, drp_col2]].dropna()

            if len(valid_drugs) >= 3:
                data1 = valid_drugs[drp_col1].values
                data2 = valid_drugs[drp_col2].values

                comparison = paired_comparison_comprehensive(
                    data1, data2,
                    strategy1.upper(), strategy2.upper()
                )

                if comparison:
                    pairwise_tests.append({
                        'Comparison': f'{strategy1.upper()} → {strategy2.upper()}',
                        'Metric': metric_name,
                        'N_Pairs': comparison['n_pairs'],
                        'Mean_Diff': comparison['mean_diff'],
                        'Median_Diff': comparison['median_diff'],
                        'Std_Diff': comparison['std_diff'],
                        'T_Statistic': comparison['t_statistic'],
                        'T_PValue': comparison['t_pvalue'],
                        'T_Significant': '✅' if comparison['t_significant'] else '❌',
                        'Wilcoxon_Statistic': comparison['wilcoxon_statistic'],
                        'Wilcoxon_PValue': comparison['wilcoxon_pvalue'],
                        'Wilcoxon_Significant': '✅' if comparison['wilcoxon_significant'] else '❌',
                        'Cohens_D': comparison['cohens_d'],
                        'Cohens_D_Interpretation': comparison['cohens_d_interpretation'],
                        'Cliffs_Delta': comparison['cliffs_delta'],
                        'Cliffs_Delta_Interpretation': comparison['cliffs_delta_interpretation'],
                        'CI_Lower': comparison['ci_lower'],
                        'CI_Upper': comparison['ci_upper'],
                        'CI_Excludes_Zero': '✅' if comparison['ci_excludes_zero'] else '❌',
                        'Win_Rate': comparison['win_rate']
                    })

    pairwise_df = pd.DataFrame(pairwise_tests) if pairwise_tests else None

    # ============================================================
    # Tier 3: Confusion Matrices
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  Tier 3: Generate Confusion Matrices")
    print(f"{'─' * 80}")

    confusion_matrices = {}
    for strategy in available_strategies:
        if len(all_predictions[strategy]['y_true_drp']) > 0:
            cm_drp = confusion_matrix(
                all_predictions[strategy]['y_true_drp'],
                all_predictions[strategy]['y_pred_drp']
            )

            confusion_matrices[strategy] = {
                'drp': cm_drp,
                'y_true_drp': np.array(all_predictions[strategy]['y_true_drp']),
                'y_pred_drp': np.array(all_predictions[strategy]['y_pred_drp'])
            }

            print(f"  {strategy.upper()} DRP Confusion Matrix Shape: {cm_drp.shape}")

    # ⭐ Save results (including Bootstrap CI)
    stats_output_path = os.path.join(batch_output_root, "Three_Tier_Statistical_Tests_With_Bootstrap_CI.xlsx")

    with pd.ExcelWriter(stats_output_path, engine='openpyxl') as writer:
        baseline_df.to_excel(writer, sheet_name='Tier1_Permutation_Bootstrap', index=False)
        worksheet = writer.sheets['Tier1_Permutation_Bootstrap']

        p_col_idx = baseline_df.columns.get_loc('P_Value') + 1  # Excel is 1-indexed

        for row_idx in range(2, len(baseline_df) + 2):  # Row 1 is header
            worksheet.cell(
                row=row_idx,
                column=p_col_idx
            ).number_format = '0.0000E+00'

        if pairwise_df is not None:
            pairwise_df.to_excel(writer, sheet_name='Tier2_Pairwise_Multi_Metrics', index=False)

        # Confusion Matrices
        for strategy, cm_data in confusion_matrices.items():
            cm_df = pd.DataFrame(
                cm_data['drp'],
                index=[f'True_{i}' for i in range(cm_data['drp'].shape[0])],
                columns=[f'Pred_{i}' for i in range(cm_data['drp'].shape[1])]
            )
            sheet_name = f'CM_{strategy.upper()}'[:31]
            cm_df.to_excel(writer, sheet_name=sheet_name)

        # ⭐ Description (Updated)
        summary_text = pd.DataFrame({
            'Tier': ['Tier 1', 'Tier 1 (cont.)', 'Tier 2', 'Tier 3'],
            'Method': [
                'Label Permutation Test (empirical null distribution)',
                '⭐ Stratified Bootstrap 95% CI (within-DRP compound resampling)',
                'Paired t-test + Wilcoxon for multiple metrics',
                'Confusion Matrix + Multi-dimensional metrics'
            ],
            'Purpose': [
                'Test significance: H0 = DRP labels no better than random',
                '⭐ Quantify precision of point estimates under fixed DRP structure',
                'Compare strategy robustness to information leakage',
                'Per-class performance evaluation'
            ],
            'Interpretation': [
                'If p < 0.05: transferability significantly exceeds chance',
                '⭐ Narrow CI → stable estimate; CI excludes null mean → robust effect',
                'Negative diff → performance drops with stricter exclusion',
                'Off-diagonal elements → systematic misclassification patterns'
            ]
        })
        summary_text.to_excel(writer, sheet_name='README', index=False)

    print(f"\n{'─' * 80}")
    print(f"  ✅ Statistical test results saved: Three_Tier_Statistical_Tests_With_Bootstrap_CI.xlsx")
    print(f"{'─' * 80}")

    return {
        'baseline_tests': baseline_df,
        'pairwise_tests': pairwise_df,
        'confusion_matrices': confusion_matrices,
        'permutation_distributions': permutation_distributions,
        'bootstrap_distributions': bootstrap_distributions,  # ⭐ NEW
        'available_strategies': available_strategies,
        'all_predictions': all_predictions
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ⭐ ENHANCED: Visualizations (Add Bootstrap CI)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def create_multi_metric_comparison_with_bootstrap_ci(batch_output_root, stats_results):
    """⭐ Visualize multi-metric comparison (including Bootstrap CI and empirical baseline)"""
    baseline_df = stats_results['baseline_tests']
    strategies = stats_results['available_strategies']

    # Filter DRP results
    drp_data = baseline_df[baseline_df['Metric'] == 'DRP'].copy()

    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'cohen_kappa']
    metric_labels = ['Accuracy\n(%)', 'Balanced\nAccuracy (%)', 'Macro-F1\n(%)', 'Cohen\'s Kappa']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        metric_data = drp_data[drp_data['Indicator'] == metric]

        x_pos = np.arange(len(strategies))
        observed_vals = []
        bootstrap_ci_lower = []
        bootstrap_ci_upper = []
        empirical_means = []
        empirical_ci_lower = []
        empirical_ci_upper = []

        for strategy in strategies:
            strategy_data = metric_data[metric_data['Strategy'] == strategy.upper()]
            if len(strategy_data) > 0:
                row = strategy_data.iloc[0]
                observed_vals.append(row['Observed'])
                bootstrap_ci_lower.append(row['Bootstrap_CI_Lower'])
                bootstrap_ci_upper.append(row['Bootstrap_CI_Upper'])
                empirical_means.append(row['Empirical_Mean'])
                empirical_ci_lower.append(row['Empirical_95CI_Lower'])
                empirical_ci_upper.append(row['Empirical_95CI_Upper'])
            else:
                observed_vals.append(0)
                bootstrap_ci_lower.append(0)
                bootstrap_ci_upper.append(0)
                empirical_means.append(0)
                empirical_ci_lower.append(0)
                empirical_ci_upper.append(0)

        # ⭐ Calculate error bars for Bootstrap CI
        bootstrap_yerr = [
            np.array(observed_vals) - np.array(bootstrap_ci_lower),
            np.array(bootstrap_ci_upper) - np.array(observed_vals)
        ]

        # ⭐ Draw bar charts with Bootstrap CI
        bars = ax.bar(x_pos, observed_vals,
                      alpha=0.75,
                      edgecolor='black',
                      linewidth=1.5,
                      color=['#9B59B6', '#2ECC71', '#E74C3C', '#F39C12', '#3498DB'][:len(strategies)],
                      label='Observed ± 95% Boot CI')

        # ⭐ Add Bootstrap CI error bars
        ax.errorbar(x_pos, observed_vals,
                   yerr=bootstrap_yerr,
                   fmt='none',
                   ecolor='black',
                   elinewidth=2,
                   capsize=5,
                   capthick=2,
                   alpha=0.8)

        # Draw empirical baseline (Permutation null mean)
        ax.plot(x_pos, empirical_means,
               'r--', linewidth=2.5, marker='o', markersize=8,
               label='Permutation Null Mean',
               zorder=10)

        # ⭐ Draw 95% CI for the empirical baseline (shaded region)
        ax.fill_between(x_pos,
                       empirical_ci_lower,
                       empirical_ci_upper,
                       alpha=0.2,
                       color='red',
                       label='Permutation Null 95% CI',
                       zorder=5)

        # Annotate significance
        for i, (strategy, bar) in enumerate(zip(strategies, bars)):
            strategy_data = metric_data[metric_data['Strategy'] == strategy.upper()]
            if len(strategy_data) > 0:
                p_val = strategy_data.iloc[0]['P_Value']

                # ⭐ Simultaneously check if Bootstrap CI overlaps with the empirical baseline
                boot_lower = bootstrap_ci_lower[i]
                emp_upper = empirical_ci_upper[i]
                ci_excludes_null = boot_lower > emp_upper

                if p_val < 0.001:
                    marker = '***'
                elif p_val < 0.01:
                    marker = '**'
                elif p_val < 0.05:
                    marker = '*'
                else:
                    marker = 'ns'

                # ⭐ Mark green if Bootstrap CI does not overlap with the empirical baseline
                color = 'green' if ci_excludes_null else 'red'

                ax.text(bar.get_x() + bar.get_width()/2,
                       bootstrap_ci_upper[i] + 2,
                       marker,
                       ha='center',
                       fontsize=12,
                       fontweight='bold',
                       color=color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.upper() for s in strategies],
                          rotation=15, ha='right', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric_label.replace(chr(10), " ")}\n(*** p<0.001, green=CI excludes null)',
                    fontweight='bold', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('⭐ Multi-Metric Performance: Observed ± Bootstrap CI vs Permutation Null Baseline',
                fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    output_path = os.path.join(batch_output_root, "Multi_Metric_With_Bootstrap_CI")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Multi-metric comparison (with Bootstrap CI) saved: Multi_Metric_With_Bootstrap_CI.pdf/.png")


def create_bootstrap_distribution_visualization(batch_output_root, stats_results):
    """⭐ Visualize Bootstrap Distribution (Compared with Permutation Distribution)"""
    perm_dists = stats_results['permutation_distributions']
    boot_dists = stats_results['bootstrap_distributions']
    strategies = stats_results['available_strategies']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = ['balanced_accuracy', 'macro_f1']
    metric_names = ['Balanced Accuracy (%)', 'Macro-F1 (%)']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # DRP Level
        ax_drp = axes[idx, 0]

        for strategy in strategies[:3]:  # Show only the first 3 strategies to avoid crowding
            if strategy not in perm_dists or strategy not in boot_dists:
                continue

            # Permutation distribution (empirical null distribution)
            perm_data = perm_dists[strategy]['drp'][metric]
            perm_dist = perm_data['permutation_distribution']

            # Bootstrap distribution (sampling uncertainty)
            boot_data = boot_dists[strategy]['drp'][metric]
            boot_dist = boot_data['bootstrap_distribution']

            observed = perm_data['observed']

            # Draw Permutation distribution (red, semi-transparent)
            ax_drp.hist(perm_dist, bins=40, alpha=0.3,
                       label=f'{strategy.upper()} Permutation Null',
                       density=True, color='red', edgecolor='darkred', linewidth=0.5)

            # ⭐ Draw Bootstrap distribution (blue, semi-transparent)
            ax_drp.hist(boot_dist, bins=40, alpha=0.3,
                       label=f'{strategy.upper()} Bootstrap',
                       density=True, color='blue', edgecolor='darkblue', linewidth=0.5)

            # Mark observed values
            ax_drp.axvline(observed, color='green', linestyle='--', linewidth=2.5,
                          label=f'{strategy.upper()} Observed={observed:.1f}')

        ax_drp.set_xlabel(f'DRP {metric_name}', fontweight='bold', fontsize=12)
        ax_drp.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax_drp.set_title(f'DRP: {metric_name}\n(Red=Null, Blue=Bootstrap, Green=Observed)',
                        fontweight='bold', fontsize=12)
        ax_drp.legend(fontsize=8, loc='upper right')
        ax_drp.grid(alpha=0.3)

        # Family Level
        ax_fam = axes[idx, 1]

        for strategy in strategies[:3]:
            if strategy not in perm_dists or strategy not in boot_dists:
                continue

            perm_data = perm_dists[strategy]['family'][metric]
            perm_dist = perm_data['permutation_distribution']

            boot_data = boot_dists[strategy]['family'][metric]
            boot_dist = boot_data['bootstrap_distribution']

            observed = perm_data['observed']

            ax_fam.hist(perm_dist, bins=40, alpha=0.3,
                       label=f'{strategy.upper()} Permutation Null',
                       density=True, color='red', edgecolor='darkred', linewidth=0.5)

            ax_fam.hist(boot_dist, bins=40, alpha=0.3,
                       label=f'{strategy.upper()} Bootstrap',
                       density=True, color='blue', edgecolor='darkblue', linewidth=0.5)

            ax_fam.axvline(observed, color='green', linestyle='--', linewidth=2.5,
                          label=f'{strategy.upper()} Observed={observed:.1f}')

        ax_fam.set_xlabel(f'Family {metric_name}', fontweight='bold', fontsize=12)
        ax_fam.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax_fam.set_title(f'Family: {metric_name}\n(Red=Null, Blue=Bootstrap, Green=Observed)',
                        fontweight='bold', fontsize=12)
        ax_fam.legend(fontsize=8, loc='upper right')
        ax_fam.grid(alpha=0.3)

    plt.suptitle('⭐ Comparison: Permutation Null vs Bootstrap Distributions',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_path = os.path.join(batch_output_root, "Bootstrap_vs_Permutation_Distributions")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Bootstrap vs Permutation distribution comparison saved: Bootstrap_vs_Permutation_Distributions.pdf/.png")


def create_confusion_matrix_visualization(batch_output_root, stats_results):
    """Visualize Confusion Matrix (Kept unchanged)"""
    confusion_matrices = stats_results['confusion_matrices']
    strategies = stats_results['available_strategies']

    n_strategies = len([s for s in strategies if s in confusion_matrices])
    if n_strategies == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies[:6]):
        if strategy not in confusion_matrices:
            axes[idx].axis('off')
            continue

        cm = confusion_matrices[strategy]['drp']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        im = axes[idx].imshow(cm_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                axes[idx].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                             ha='center', va='center', color=text_color, fontsize=8)

        y_true = confusion_matrices[strategy]['y_true_drp']
        y_pred = confusion_matrices[strategy]['y_pred_drp']
        acc = np.mean(y_true == y_pred) * 100
        bal_acc = balanced_accuracy_score(y_true, y_pred) * 100

        axes[idx].set_title(f'{strategy.upper()}\nAcc={acc:.1f}%, Bal-Acc={bal_acc:.1f}%',
                          fontweight='bold', fontsize=11)
        axes[idx].set_xlabel('Predicted DRP', fontweight='bold')
        axes[idx].set_ylabel('True DRP', fontweight='bold')

        axes[idx].set_xticks(range(cm.shape[1]))
        axes[idx].set_yticks(range(cm.shape[0]))
        axes[idx].set_xticklabels(range(cm.shape[1]), fontsize=8)
        axes[idx].set_yticklabels(range(cm.shape[0]), fontsize=8)

    for idx in range(n_strategies, 6):
        axes[idx].axis('off')

    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05,
                 label='Normalized Frequency (Row-wise)', fraction=0.046)

    plt.suptitle('Confusion Matrices Across Strategies (DRP Classification)',
                fontsize=14, fontweight='bold', y=0.98)

    output_path = os.path.join(batch_output_root, "Confusion_Matrices")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Confusion matrix visualization saved: Confusion_Matrices.pdf/.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main Program                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    """Batch processing main program"""
    os.makedirs(BATCH_OUTPUT_ROOT, exist_ok=True)

    print("╔" + "═" * 80 + "╗")
    print("║" + "  ENHANCED BENCHMARK: MULTI-METRICS + PERMUTATION + ⭐STRATIFIED BOOTSTRAP".center(80) + "║")
    print("╚" + "═" * 80 + "╝")

    grouping_info = load_compound_grouping_info()

    if grouping_info['compound_annotation'] is None:
        strategies = ['no_exclusion', 'identity']
    else:
        strategies = ['no_exclusion', 'identity', 'api', 'chemical', 'moa']

    print(f"\n  Enabled strategies: {strategies}")
    print(f"  ⭐ New Metrics: Balanced Accuracy, Macro-F1, Cohen's Kappa")
    print(f"  ⭐ New Method: Label permutation empirical baseline (n={N_PERMUTATIONS})")
    print(f"  ⭐⭐ New Method: Stratified Bootstrap 95% CI (n={N_BOOTSTRAP})")

    # Read reference data
    try:
        ref_data = pd.read_csv(REF_MATRIX_PATH, index_col=0)
        umap_df = pd.read_csv(UMAP_COORD_PATH)
        label_df = pd.read_csv(LABEL_PATH)

        id_col_umap = umap_df.columns[0]
        u1_col = [c for c in umap_df.columns if 'umap1' in c.lower()][0]
        u2_col = [c for c in umap_df.columns if 'umap2' in c.lower()][0]
        id_col_label = label_df.columns[0]
        cl_col = [c for c in label_df.columns if any(k in c.lower() for k in ['cluster', 'label'])][0]

        ref_info = umap_df[[id_col_umap, u1_col, u2_col]].merge(
            label_df[[id_col_label, cl_col]],
            left_on=id_col_umap, right_on=id_col_label, how='inner'
        )
        ref_info = ref_info.rename(columns={
            id_col_umap: 'Drug', u1_col: 'UMAP1', u2_col: 'UMAP2', cl_col: 'Cluster'
        })
        ref_info = ref_info[['Drug', 'UMAP1', 'UMAP2', 'Cluster']]
        ref_info['Cluster'] = ref_info['Cluster'].astype(int)

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return

    # Batch processing
    drug_folders = [d for d in os.listdir(DRUG_PROFILES_ROOT)
                    if os.path.isdir(os.path.join(DRUG_PROFILES_ROOT, d))]
    drug_folders = sorted(drug_folders)

    processing_summary = []
    failed_drugs = []

    for i, drug_folder_name in enumerate(drug_folders, 1):
        print(f"\n[{i}/{len(drug_folders)}] {drug_folder_name}")

        drug_folder_path = os.path.join(DRUG_PROFILES_ROOT, drug_folder_name)
        drug_output_dir = os.path.join(BATCH_OUTPUT_ROOT, drug_folder_name)
        os.makedirs(drug_output_dir, exist_ok=True)

        try:
            summary = process_single_drug_with_exclusion(
                drug_name=drug_folder_name,
                drug_folder=drug_folder_path,
                output_dir=drug_output_dir,
                ref_data=ref_data,
                umap_df=umap_df,
                label_df=label_df,
                ref_info=ref_info,
                grouping_info=grouping_info,
                strategies=strategies
            )

            summary['Status'] = 'Success'
            processing_summary.append(summary)

        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            traceback.print_exc()
            failed_drugs.append({
                'Drug': drug_folder_name,
                'Status': 'Failed',
                'Error_Message': str(e)
            })

    # Save summary
    all_summaries = processing_summary + failed_drugs
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(BATCH_OUTPUT_ROOT, "Batch_Processing_Summary_With_Bootstrap.csv"), index=False)

    # ⭐ Three-Tier Statistical Tests (including Stratified Bootstrap)
    stats_results = perform_three_tier_statistical_tests(BATCH_OUTPUT_ROOT, summary_df)

    if stats_results:
        # Generate visualizations
        print(f"\n{'═' * 80}")
        print(f"  Generate Visualizations")
        print(f"{'═' * 80}")

        create_confusion_matrix_visualization(BATCH_OUTPUT_ROOT, stats_results)
        create_multi_metric_comparison_with_bootstrap_ci(BATCH_OUTPUT_ROOT, stats_results)  # ⭐ New version
        create_bootstrap_distribution_visualization(BATCH_OUTPUT_ROOT, stats_results)  # ⭐ New

    # Final Report
    print(f"\n")
    print("╔" + "═" * 80 + "╗")
    print("║" + "  Processing Complete".center(80) + "║")
    print("╠" + "═" * 80 + "╣")
    print(f"║  Success: {len(processing_summary)} | Failed: {len(failed_drugs)}".ljust(81) + "║")
    print(f"║  Output Files:".ljust(81) + "║")
    print(f"║    - Batch_Processing_Summary_With_Bootstrap.csv".ljust(81) + "║")
    print(f"║    - Three_Tier_Statistical_Tests_With_Bootstrap_CI.xlsx ⭐⭐".ljust(81) + "║")
    print(f"║    - Confusion_Matrices.pdf/.png".ljust(81) + "║")
    print(f"║    - Multi_Metric_With_Bootstrap_CI.pdf/.png ⭐⭐".ljust(81) + "║")
    print(f"║    - Bootstrap_vs_Permutation_Distributions.pdf/.png ⭐⭐".ljust(81) + "║")
    print(f"║".ljust(81) + "║")
    print(f"║  ⭐⭐ New Features:".ljust(81) + "║")
    print(f"║    ✅ Stratified Bootstrap: Resample compounds within each reference DRP".ljust(81) + "║")
    print(f"║    ✅ Each compound carries all its external profiles".ljust(81) + "║")
    print(f"║    ✅ Ensure every bootstrap sample contains all DRP classes".ljust(81) + "║")
    print(f"║    ✅ Visualization simultaneously displays: Observed ± Bootstrap CI vs Permutation Null".ljust(81) + "║")
    print(f"║    ✅ Green marker: Bootstrap CI does not overlap with empirical baseline (robust effect)".ljust(81) + "║")
    print("╚" + "═" * 80 + "╝")


if __name__ == "__main__":
    main()
