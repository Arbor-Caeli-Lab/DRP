"""
═══════════════════════════════════════════════════════════════════════════════
  Leave-One-Out Cross-Validation (LOO) for Drug Projection Analysis
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cdist
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  0. Configuration Parameters                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Reference Data Paths (Do not change)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter06"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter06"
REF_MATRIX_PATH = DATA_ROOT / "01_HepG2_367_Drugs_log2FC_6682_for_UMAP.csv"
UMAP_COORD_PATH = DATA_ROOT / "HepG2_coordinate system.csv"
LABEL_PATH = DATA_ROOT / "HepG2_label.csv"

# Output Directory for LOO Validation
LOO_OUTPUT_ROOT = RESULTS_ROOT / "01_367compounds_Leave-one-out"

# DRP Configuration
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

DRP_COLORS = {
    1: "#E41A1C", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#FF7F00",
    6: "#A65628", 7: "#F781BF", 8: "#808080", 9: "#66C2A5", 10: "#E6AB02",
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. Leave-One-Out Cross-Validation                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def leave_one_out_validation(ref_data, ref_info, k_values=[3, 5, 7, 10, 15, 20],
                             use_adaptive_k=False, similarity_threshold=0.5):
    """
    Leave-One-Out Cross-Validation
    """
    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  LEAVE-ONE-OUT CROSS-VALIDATION".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║  Reference Drugs: {len(ref_data)}".ljust(79) + "║")
    print(f"║  Proteins: {ref_data.shape[1]}".ljust(79) + "║")
    print(f"║  Test K Values: {k_values}".ljust(79) + "║")
    print("╚" + "═" * 78 + "╝\n")

    start_time = time.time()
    ref_drugs = list(ref_data.index)
    n_drugs = len(ref_drugs)

    # Build fast query dictionary
    ref_info_dict = ref_info.set_index('Drug').to_dict('index')

    # Standardize data (done once)
    scaler = StandardScaler()
    ref_std = scaler.fit_transform(ref_data.values)

    all_results = []

    print("  Starting Leave-One-Out validation...")
    print("  " + "─" * 70)

    # LOO Main Loop
    for i, test_drug in enumerate(ref_drugs):
        # Display progress
        if (i + 1) % 50 == 0 or i == 0 or i == n_drugs - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_drugs - i - 1) / speed if speed > 0 else 0
            print(f"  [{i + 1:3d}/{n_drugs}] {test_drug[:30]:30s} | "
                  f"Speed: {speed:.1f} drugs/s | ETA: {eta:.0f}s")

        # Get true labels
        true_info = ref_info_dict[test_drug]
        true_umap1 = true_info['UMAP1']
        true_umap2 = true_info['UMAP2']
        true_cluster = int(true_info['Cluster'])

        # Leave one out: Build training set
        train_mask = np.ones(n_drugs, dtype=bool)
        train_mask[i] = False

        ref_std_train = ref_std[train_mask]
        test_std = ref_std[i:i + 1]

        ref_drugs_train = [ref_drugs[j] for j in range(n_drugs) if train_mask[j]]

        # Extract UMAP coordinates and labels for training set
        ref_umap1_train = np.array([ref_info_dict[d]['UMAP1'] for d in ref_drugs_train])
        ref_umap2_train = np.array([ref_info_dict[d]['UMAP2'] for d in ref_drugs_train])
        ref_clusters_train = np.array([ref_info_dict[d]['Cluster'] for d in ref_drugs_train])

        # Calculate similarity
        cos_sim = cosine_similarity(test_std, ref_std_train)[0]

        # Test each K value
        for k in k_values:
            adaptive_k = max(3, min(k, len(cos_sim[cos_sim > similarity_threshold]))) if use_adaptive_k else k

            topk_idx = np.argsort(cos_sim)[::-1][:adaptive_k]
            k_drugs = [ref_drugs_train[j] for j in topk_idx]
            k_sims = cos_sim[topk_idx]
            k_clusters = ref_clusters_train[topk_idx]
            k_u1 = ref_umap1_train[topk_idx]
            k_u2 = ref_umap2_train[topk_idx]

            # Method 1: Weighted Voting for DRP Assignment
            w = np.maximum(k_sims, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(adaptive_k) / adaptive_k

            cluster_scores = {}
            for cl, wi in zip(k_clusters, w):
                cluster_scores[cl] = cluster_scores.get(cl, 0) + wi

            pred_cluster = max(cluster_scores, key=cluster_scores.get)
            confidence = cluster_scores[pred_cluster]

            # Method 2: Weighted Average for UMAP Coordinates
            pred_u1 = np.sum(w * k_u1)
            pred_u2 = np.sum(w * k_u2)

            # Evaluate
            position_error = np.sqrt((pred_u1 - true_umap1) ** 2 + (pred_u2 - true_umap2) ** 2)
            drp_correct = (pred_cluster == true_cluster)
            top1_correct = (k_clusters[0] == true_cluster)
            topk_correct = (true_cluster in k_clusters)

            # Save results
            all_results.append({
                'Drug': test_drug,
                'K': k,
                'True_UMAP1': round(true_umap1, 4),
                'True_UMAP2': round(true_umap2, 4),
                'Pred_UMAP1': round(pred_u1, 4),
                'Pred_UMAP2': round(pred_u2, 4),
                'True_DRP': true_cluster,
                'Pred_DRP': int(pred_cluster),
                'DRP_Correct': drp_correct,
                'Top1_DRP': int(k_clusters[0]),
                'Top1_Correct': top1_correct,
                'TopK_Correct': topk_correct,
                'Position_Error': round(position_error, 4),
                'Confidence': round(confidence, 4),
                'Top1_Similarity': round(k_sims[0], 4),
                'Top1_Drug': k_drugs[0],
                'TopK_Drugs': '|'.join(k_drugs),
                'TopK_Similarities': '|'.join([f"{s:.3f}" for s in k_sims])
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate summary statistics
    print("\n  " + "─" * 70)
    print("  Calculating summary statistics...")
    summary_data = []

    for k in k_values:
        k_results = results_df[results_df['K'] == k]

        drp_accuracy = k_results['DRP_Correct'].mean() * 100
        top1_accuracy = k_results['Top1_Correct'].mean() * 100
        topk_accuracy = k_results['TopK_Correct'].mean() * 100
        mean_pos_error = k_results['Position_Error'].mean()
        mean_confidence = k_results['Confidence'].mean()

        # Per-DRP Accuracy (1 to 10)
        drp_accuracies = []
        for drp in range(1, 11):
            drp_mask = k_results['True_DRP'] == drp
            if drp_mask.sum() > 0:
                drp_acc = k_results.loc[drp_mask, 'DRP_Correct'].mean() * 100
                drp_accuracies.append(drp_acc)
            else:
                drp_accuracies.append(np.nan)

        summary_data.append({
            'K': k,
            'DRP_Accuracy': round(drp_accuracy, 2),
            'Top1_Accuracy': round(top1_accuracy, 2),
            'TopK_Accuracy': round(topk_accuracy, 2),
            'Mean_Position_Error': round(mean_pos_error, 4),
            'Mean_Confidence': round(mean_confidence, 4),
            **{f'DRP{i+1}_Accuracy': round(acc, 2) if not np.isnan(acc) else np.nan
               for i, acc in enumerate(drp_accuracies)}
        })

    summary_df = pd.DataFrame(summary_data)
    total_time = time.time() - start_time

    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  LOO VALIDATION RESULTS SUMMARY".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(summary_df[['K', 'DRP_Accuracy', 'Top1_Accuracy', 'TopK_Accuracy',
                      'Mean_Position_Error', 'Mean_Confidence']].to_string(index=False))
    print("╠" + "═" * 78 + "╣")

    best_k_idx = summary_df['DRP_Accuracy'].idxmax()
    best_k = int(summary_df.loc[best_k_idx, 'K'])

    print(f"║  🏆 Optimal K Value: {best_k}".ljust(79) + "║")
    print(f"║     - DRP Accuracy: {summary_df.loc[best_k_idx, 'DRP_Accuracy']:.2f}%".ljust(79) + "║")
    print(f"║     - Position Error: {summary_df.loc[best_k_idx, 'Mean_Position_Error']:.4f}".ljust(79) + "║")
    print(f"║     - Total Time: {total_time:.1f}s ({total_time / 60:.1f} min)".ljust(79) + "║")
    print("╚" + "═" * 78 + "╝\n")

    return results_df, summary_df, best_k

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. LOO Results Visualization                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def visualize_loo_results(results_df, summary_df, output_dir):
    """
    Generate comprehensive visualization report for LOO validation
    """
    print("\n  Generating LOO Visualizations...")

    best_k = int(summary_df.loc[summary_df['DRP_Accuracy'].idxmax(), 'K'])
    best_results = results_df[results_df['K'] == best_k].copy()

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # ============ Plot 1: Projection Position ============
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    correct_mask = best_results['DRP_Correct'].values

    for drp in range(1, 11):
        drp_mask = best_results['True_DRP'] == drp
        if drp_mask.sum() > 0:
            ax1.scatter(best_results.loc[drp_mask, 'True_UMAP1'],
                        best_results.loc[drp_mask, 'True_UMAP2'],
                        c=DRP_COLORS[drp], s=150, alpha=0.4,
                        edgecolors=DRP_COLORS[drp], linewidth=2.5,
                        label=f'C{drp} (n={drp_mask.sum()})',
                        marker='o', facecolors='none', zorder=2)

    ax1.scatter(best_results.loc[correct_mask, 'Pred_UMAP1'],
                best_results.loc[correct_mask, 'Pred_UMAP2'],
                c='green', s=80, alpha=0.8, marker='x', linewidth=3,
                label=f'✓ Correct ({correct_mask.sum()})', zorder=5)

    ax1.scatter(best_results.loc[~correct_mask, 'Pred_UMAP1'],
                best_results.loc[~correct_mask, 'Pred_UMAP2'],
                c='red', s=100, alpha=0.9, marker='X', linewidth=3,
                label=f'✗ Wrong ({(~correct_mask).sum()})', zorder=6)

    for idx, row in best_results.iterrows():
        color = 'green' if row['DRP_Correct'] else 'red'
        ax1.plot([row['True_UMAP1'], row['Pred_UMAP1']],
                 [row['True_UMAP2'], row['Pred_UMAP2']],
                 color=color, alpha=0.2 if row['DRP_Correct'] else 0.5,
                 linewidth=0.5 if row['DRP_Correct'] else 1.5, zorder=1)

    ax1.set_xlabel('UMAP 1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=13, fontweight='bold')
    ax1.set_title(f'Leave-One-Out Projection Results (K={best_k})\n'
                  f'Accuracy: {correct_mask.mean() * 100:.2f}% | '
                  f'Error: {best_results["Position_Error"].mean():.3f}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2, loc='best')
    ax1.grid(alpha=0.2)

    # ============ Plot 2: Performance vs K ============
    ax2 = fig.add_subplot(gs[0, 2])
    k_vals = summary_df['K'].values
    ax2.plot(k_vals, summary_df['DRP_Accuracy'], 'o-', label='DRP Accuracy', linewidth=3, color='#E41A1C')
    ax2.plot(k_vals, summary_df['Top1_Accuracy'], 's-', label='Top1 Accuracy', linewidth=3, color='#377EB8')
    ax2.axvline(k_vals[summary_df['DRP_Accuracy'].idxmax()], color='red', linestyle='--', label=f'Best K={best_k}')
    ax2.set_title('Performance vs K Value', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    # ============ Plot 3: Error Distribution ============
    ax3 = fig.add_subplot(gs[0, 3])
    errors = best_results['Position_Error']
    ax3.hist(errors, bins=40, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.3f}')
    ax3.set_title('Projection Error Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # ============ Plot 4: Confusion Matrix ============
    ax4 = fig.add_subplot(gs[1, 2:4])
    cm = confusion_matrix(best_results['True_DRP'], best_results['Pred_DRP'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=range(1, 11), yticklabels=range(1, 11),
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    ax4.set_xlabel('Predicted DRP', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True DRP', fontsize=12, fontweight='bold')
    ax4.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    for i in range(10):
        total = cm[i, :].sum()
        if total > 0:
            ax4.text(i + 0.5, i - 0.3, f'{cm[i, i]/total*100:.0f}%',
                     ha='center', va='center', fontsize=9, fontweight='bold', color='darkgreen')

    # ============ Plot 5: Per-DRP Accuracy ============
    ax5 = fig.add_subplot(gs[2, 0])
    drp_accuracy, drp_counts = [], []
    for drp in range(1, 11):
        drp_mask = best_results['True_DRP'] == drp
        drp_accuracy.append(best_results.loc[drp_mask, 'DRP_Correct'].mean() * 100 if drp_mask.sum() > 0 else 0)
        drp_counts.append(drp_mask.sum())

    bars = ax5.bar(range(1, 11), drp_accuracy, color=[DRP_COLORS[i] for i in range(1, 11)], edgecolor='black')
    for bar, count in zip(bars, drp_counts):
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2, f'n={count}', ha='center', va='bottom', fontsize=8)
    ax5.axhline(best_results['DRP_Correct'].mean() * 100, color='red', linestyle='--')
    ax5.set_title('Per-DRP Classification Accuracy', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(1, 11))
    ax5.set_ylim([0, 110])

    # ============ Plot 6: Similarity vs Accuracy ============
    ax6 = fig.add_subplot(gs[2, 1])
    sim_bins = np.arange(0.3, 1.0, 0.05)
    bin_centers, bin_accuracies = [], []
    for i in range(len(sim_bins) - 1):
        mask = (best_results['Top1_Similarity'] >= sim_bins[i]) & (best_results['Top1_Similarity'] < sim_bins[i + 1])
        if mask.sum() > 5:
            bin_centers.append((sim_bins[i] + sim_bins[i + 1]) / 2)
            bin_accuracies.append(best_results.loc[mask, 'DRP_Correct'].mean() * 100)
    ax6.plot(bin_centers, bin_accuracies, 'o-', linewidth=2.5, color='purple')
    ax6.set_title('Accuracy vs Similarity', fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.grid(alpha=0.3)

    # ============ Plot 7: Error by Correctness ============
    ax7 = fig.add_subplot(gs[2, 2])
    bp = ax7.boxplot([best_results.loc[correct_mask, 'Position_Error'], best_results.loc[~correct_mask, 'Position_Error']],
                     labels=['Correct', 'Wrong'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax7.set_title('Error Distribution by Correctness', fontsize=13, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)

    # Overall Title and Save
    plt.suptitle(f'Leave-One-Out Cross-Validation Summary Report (K={best_k})\n'
                 f'Overall Accuracy: {best_results["DRP_Correct"].mean() * 100:.2f}%',
                 fontsize=16, fontweight='bold', y=0.995)

    output_path = os.path.join(output_dir, 'LOO_Validation_Report')
    plt.savefig(output_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✅ LOO Visualizations Saved: {output_path}.pdf/.png")

    # Generate Classification Report
    report = classification_report(best_results['True_DRP'], best_results['Pred_DRP'],
                                   target_names=[f'DRP-{i}' for i in range(1, 11)], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, 'LOO_Classification_Report.csv'))
    print(f"    ✅ Classification Report Saved: LOO_Classification_Report.csv")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. Quick LOO Check (Optional)                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def quick_loo_check(ref_data, ref_info, k=5, n_samples=50):
    import random
    print(f"\n  Quick LOO Check (K={k}, Samples={n_samples})...")
    ref_drugs = list(ref_data.index)
    sample_drugs = random.sample(ref_drugs, min(n_samples, len(ref_drugs)))
    scaler = StandardScaler()
    ref_std = scaler.fit_transform(ref_data.values)
    ref_info_dict = ref_info.set_index('Drug').to_dict('index')

    correct_drp = 0

    for test_drug in sample_drugs:
        test_idx = ref_drugs.index(test_drug)
        true_cluster = ref_info_dict[test_drug]['Cluster']

        train_mask = np.ones(len(ref_drugs), dtype=bool)
        train_mask[test_idx] = False
        ref_std_train = ref_std[train_mask]
        test_std = ref_std[test_idx:test_idx + 1]

        ref_drugs_train = [d for d in ref_drugs if d != test_drug]
        ref_clusters_train = np.array([ref_info_dict[d]['Cluster'] for d in ref_drugs_train])

        cos_sim = cosine_similarity(test_std, ref_std_train)[0]
        topk_idx = np.argsort(cos_sim)[::-1][:k]
        k_clusters = ref_clusters_train[topk_idx]
        k_sims = cos_sim[topk_idx]

        w = k_sims / k_sims.sum()
        cluster_scores = {}
        for cl, wi in zip(k_clusters, w):
            cluster_scores[cl] = cluster_scores.get(cl, 0) + wi
        pred_cluster = max(cluster_scores, key=cluster_scores.get)

        if pred_cluster == true_cluster:
            correct_drp += 1

    drp_acc = correct_drp / len(sample_drugs) * 100
    print(f"  Quick Check Accuracy: {drp_acc:.2f}%\n")
    return drp_acc

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. Main Execution Function                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    # Create Output Directory
    os.makedirs(LOO_OUTPUT_ROOT, exist_ok=True)

    print("╔" + "═" * 80 + "╗")
    print("║" + "  ONLY LOO VALIDATION ON HEPG2 REFERENCE DATA".center(80) + "║")
    print("╚" + "═" * 80 + "╝\n")

    # Step 1: Load Data
    print("📂 Loading Reference Data...")
    try:
        ref_data = pd.read_csv(REF_MATRIX_PATH, index_col=0)
        umap_df = pd.read_csv(UMAP_COORD_PATH)
        label_df = pd.read_csv(LABEL_PATH)

        id_col_umap = umap_df.columns[0]
        u1_col = [c for c in umap_df.columns if 'umap' in c.lower() and '1' in c][0]
        u2_col = [c for c in umap_df.columns if 'umap' in c.lower() and '2' in c][0]

        id_col_label = label_df.columns[0]
        cl_col = [c for c in label_df.columns if any(k in c.lower() for k in ['cluster', 'label', 'group'])][0]

        ref_info = umap_df[[id_col_umap, u1_col, u2_col]].merge(
            label_df[[id_col_label, cl_col]], left_on=id_col_umap, right_on=id_col_label, how='inner'
        )
        ref_info = ref_info.rename(columns={id_col_umap: 'Drug', u1_col: 'UMAP1', u2_col: 'UMAP2', cl_col: 'Cluster'})
        ref_info['Cluster'] = ref_info['Cluster'].astype(int)

        print(f"✅ Data loaded. Drugs: {len(ref_info)}, Proteins: {ref_data.shape[1]}")
    except Exception as e:
        print(f"❌ Failed to load reference data: {e}")
        return

    # Step 2: LOO Validation
    print("\n" + "=" * 80)
    print("  EXECUTING LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 80)

    # Optional quick check
    quick_acc = quick_loo_check(ref_data, ref_info, k=5, n_samples=50)

    # Full Validation
    loo_results, loo_summary, optimal_k = leave_one_out_validation(
        ref_data=ref_data,
        ref_info=ref_info,
        k_values=[3, 5, 7, 10, 15, 20],
        use_adaptive_k=False
    )

    # Step 3: Save and Visualize
    loo_results.to_csv(os.path.join(LOO_OUTPUT_ROOT, "LOO_detailed_results.csv"), index=False)
    loo_summary.to_csv(os.path.join(LOO_OUTPUT_ROOT, "LOO_summary.csv"), index=False)
    print(f"\n  ✅ Data saved to: {LOO_OUTPUT_ROOT}")

    visualize_loo_results(loo_results, loo_summary, LOO_OUTPUT_ROOT)

    print("\n" + "═" * 80)
    print(f"🎉 LOO Validation Finished Successfully! Optimal K = {optimal_k}")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    main()
