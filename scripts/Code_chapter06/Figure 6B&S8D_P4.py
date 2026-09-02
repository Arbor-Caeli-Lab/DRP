# -*- coding: utf-8 -*-

"""
===============================================================================
Drug Representation Comparison V4
Strict Identity-Out Retrieval with Pre-computed NES Matrix

Core Principles
--------
For Query Drug = X:

    External X / condition
            |
            v
    representation / similarity
            |
            v
    HepG2 reference candidates
            |
            v
        REMOVE X
            |
            v
       Rank remaining drugs



===============================================================================
"""

import os
import hashlib
import traceback
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Code_chapter06"
RESULTS_ROOT = REPO_ROOT / "results" / "Code_chapter06"

PROJECTION_RESULTS_ROOT = DATA_ROOT / "17_drug_profiles"

DRUG_PROFILES_ROOT = DATA_ROOT / "drug_profiles"

HEPG2_PROFILE_PATH = DATA_ROOT / "01_HepG2_367_Drugs_log2FC_6682_for_UMAP.csv"

REF_UMAP_PATH = DATA_ROOT / "HepG2_coordinate system.csv"

REF_LABEL_PATH = DATA_ROOT / "HepG2_label.csv"

CHEMICAL_SIMILARITY_PATH = DATA_ROOT / "tanimoto_similarity_matrix.csv"

DRUG_ANNOTATION_PATH = DATA_ROOT / "Table S1-Compound information.csv"

#
NES_MATRIX_PATH = DATA_ROOT / "NES_matrix.csv"

OUTPUT_DIR = RESULTS_ROOT / "04_MOA retrieval performance"

os.makedirs(OUTPUT_DIR, exist_ok=True)

K_VALUES = [1, 3, 5, 10]

MOA_DIMENSIONS = [
    "Subclass",
    "Primary_Target",
    "Pathways",
    "Class",
    "DRP"
]

RANDOM_SEED = 1000


# =============================================================================
# Utility Functions
# =============================================================================

def safe_read_csv(file_path, **kwargs):
    encodings = [
        "utf-8",
        "utf-8-sig",
        "gbk",
        "gb2312",
        "gb18030",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                **kwargs
            )
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Failed to read file: {file_path}\n"
        f"Last error: {last_error}"
    )


def create_logger():
    logger = logging.getLogger(
        "drug_representation_identity_out"
    )

    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    log_path = os.path.join(
        OUTPUT_DIR,
        "identity_out_comparison.log"
    )

    fh = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8"
    )

    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def remove_query_identity(drugs, query_drug, top_k=None):
    """
    Unified Identity-Out utility function.

    Prevents:
    - The query drug itself from appearing in the returned results
    - Duplicate drugs
    """

    output = []
    seen = set()

    for drug in drugs:

        if drug == query_drug:
            continue

        if drug in seen:
            continue

        seen.add(drug)
        output.append(drug)

        if top_k is not None and len(output) >= top_k:
            break

    return output


def assert_identity_out(
        retrieved_drugs,
        query_drug,
        method_name=""
):
    """
    Strictly check for identity leakage.
    """

    if query_drug in retrieved_drugs:
        raise RuntimeError(
            f"Identity leakage detected!\n"
            f"Method: {method_name}\n"
            f"Query drug: {query_drug}\n"
            f"Retrieved: {retrieved_drugs}"
        )


# =============================================================================
# 🆕 NES Matrix Diagnosis and Analysis
# =============================================================================

def diagnose_nes_matrix(nes_matrix, hepg2_coords):
    """Diagnose NES matrix quality"""

    print("\n🔍 NES Matrix Quality Diagnosis")
    print("=" * 50)

    if nes_matrix is None:
        print("❌ NES matrix not loaded")
        return False

    # Basic statistics
    print(f"Matrix dimensions: {nes_matrix.shape}")
    print(f"Missing value ratio: {nes_matrix.isnull().sum().sum() / nes_matrix.size:.2%}")
    print(f"Value range: [{nes_matrix.min().min():.3f}, {nes_matrix.max().max():.3f}]")

    # Overlap with HepG2 analysis
    hepg2_drugs = set(hepg2_coords["Drug"].astype(str))
    nes_drugs = set(nes_matrix.index)
    overlap = hepg2_drugs & nes_drugs

    print(f"Overlap with HepG2: {len(overlap)}/{len(hepg2_drugs)} ({len(overlap) / len(hepg2_drugs):.1%})")

    if len(overlap) < len(hepg2_drugs) * 0.5:
        print("⚠️  Overlap is too low; this may affect Pathway method performance")
        return False

    # Similarity distribution analysis
    sample_drugs = list(overlap)[:10]  # Take 10 overlapping drug samples
    similarities = []

    for i in range(len(sample_drugs)):
        for j in range(i + 1, len(sample_drugs)):
            drug1, drug2 = sample_drugs[i], sample_drugs[j]
            nes1 = nes_matrix.loc[drug1]
            nes2 = nes_matrix.loc[drug2]

            corr, _ = spearmanr(nes1, nes2)
            if not np.isnan(corr):
                similarities.append(corr)

    if similarities:
        similarities = np.array(similarities)
        print(f"Similarity distribution: mean={similarities.mean():.3f}, std={similarities.std():.3f}")

        if similarities.std() < 0.1:
            print("⚠️  Similarity variance is too low, may lack discriminative power")

    print("✅ NES matrix diagnosis complete")
    return True


# =============================================================================
# Reference Data (Modified: Loading NES matrix)
# =============================================================================

def load_reference_data():
    print("\n📂 Loading reference data...")

    # -------------------------------------------------------------------------
    # HepG2 protein profiles
    # -------------------------------------------------------------------------

    hepg2_profiles = safe_read_csv(
        HEPG2_PROFILE_PATH,
        index_col=0
    )

    hepg2_profiles.index = (
        hepg2_profiles.index.astype(str)
    )

    print(
        f"  ✓ HepG2 profiles: "
        f"{hepg2_profiles.shape}"
    )

    # -------------------------------------------------------------------------
    # UMAP coordinates
    # -------------------------------------------------------------------------

    umap_df = safe_read_csv(
        REF_UMAP_PATH
    )

    label_df = safe_read_csv(
        REF_LABEL_PATH
    )

    id_col_umap = umap_df.columns[0]

    u1_candidates = [
        c
        for c in umap_df.columns
        if any(
            k in c.lower()
            for k in [
                "umap1",
                "umap_1",
                "umap 1"
            ]
        )
    ]

    u2_candidates = [
        c
        for c in umap_df.columns
        if any(
            k in c.lower()
            for k in [
                "umap2",
                "umap_2",
                "umap 2"
            ]
        )
    ]

    u1_col = (
        u1_candidates[0]
        if u1_candidates
        else umap_df.columns[1]
    )

    u2_col = (
        u2_candidates[0]
        if u2_candidates
        else umap_df.columns[2]
    )

    id_col_label = label_df.columns[0]

    cluster_candidates = [
        c
        for c in label_df.columns
        if any(
            k in c.lower()
            for k in [
                "cluster",
                "label"
            ]
        )
    ]

    cluster_col = (
        cluster_candidates[0]
        if cluster_candidates
        else label_df.columns[1]
    )

    hepg2_coords = (
        umap_df[
            [
                id_col_umap,
                u1_col,
                u2_col
            ]
        ]
        .merge(
            label_df[
                [
                    id_col_label,
                    cluster_col
                ]
            ],
            left_on=id_col_umap,
            right_on=id_col_label,
            how="inner"
        )
    )

    hepg2_coords = (
        hepg2_coords.rename(
            columns={
                id_col_umap: "Drug",
                u1_col: "UMAP1",
                u2_col: "UMAP2",
                cluster_col: "DRP"
            }
        )[
            [
                "Drug",
                "UMAP1",
                "UMAP2",
                "DRP"
            ]
        ]
    )

    hepg2_coords["Drug"] = (
        hepg2_coords["Drug"].astype(str)
    )

    hepg2_coords["DRP"] = pd.to_numeric(
        hepg2_coords["DRP"],
        errors="coerce"
    )

    print(
        f"  ✓ HepG2 UMAP: "
        f"{len(hepg2_coords)} drugs"
    )

    # -------------------------------------------------------------------------
    # Drug annotation
    # -------------------------------------------------------------------------

    drug_annotations = None

    if os.path.exists(
            DRUG_ANNOTATION_PATH
    ):

        try:

            drug_annotations = safe_read_csv(
                DRUG_ANNOTATION_PATH
            )

            if (
                    "Compound_Name"
                    in drug_annotations.columns
            ):

                drug_annotations = (
                    drug_annotations.set_index(
                        "Compound_Name"
                    )
                )

            else:

                drug_annotations = (
                    drug_annotations.set_index(
                        drug_annotations.columns[0]
                    )
                )

            drug_annotations.index = (
                drug_annotations.index.astype(str)
            )

            print(
                f"  ✓ Drug annotations: "
                f"{len(drug_annotations)}"
            )

        except Exception as e:

            print(
                f"  ⚠ Drug annotation error: {e}"
            )

    # -------------------------------------------------------------------------
    # 🆕 Load Pre-computed NES Matrix (replaces dynamically calculated features)
    # -------------------------------------------------------------------------

    nes_matrix = None
    if os.path.exists(NES_MATRIX_PATH):
        try:
            print("  🧬 Loading pre-computed NES matrix...")

            nes_matrix = safe_read_csv(NES_MATRIX_PATH, index_col=0)

            # Ensure indices are string types
            nes_matrix.index = nes_matrix.index.astype(str)
            nes_matrix.columns = nes_matrix.columns.astype(str)

            # Convert to numeric types
            nes_matrix = nes_matrix.apply(pd.to_numeric, errors="coerce")

            print(f"  ✓ NES matrix: {nes_matrix.shape}")
            print(f"  ✓ Pathways: {nes_matrix.shape[1]}")
            print(f"  ✓ Drugs in NES: {nes_matrix.shape[0]}")

            # Check overlap with HepG2
            hepg2_drugs = set(hepg2_coords["Drug"].astype(str))
            nes_drugs = set(nes_matrix.index)
            overlap = hepg2_drugs & nes_drugs

            print(f"  ✓ HepG2-NES overlap: {len(overlap)}/{len(hepg2_drugs)} drugs")

            if len(overlap) < len(hepg2_drugs) * 0.8:
                print(f"  ⚠️  Warning: Low overlap between NES matrix and HepG2 ({len(overlap) / len(hepg2_drugs):.1%})")

        except Exception as e:
            print(f"  ❌ NES matrix loading failed: {e}")
            nes_matrix = None
    else:
        print(f"  ❌ NES matrix file not found: {NES_MATRIX_PATH}")

    # -------------------------------------------------------------------------
    # Chemical similarities
    # -------------------------------------------------------------------------

    chemical_similarity = None

    if os.path.exists(
            CHEMICAL_SIMILARITY_PATH
    ):

        try:

            chemical_similarity = safe_read_csv(
                CHEMICAL_SIMILARITY_PATH,
                index_col=0
            )

            chemical_similarity.index = (
                chemical_similarity
                .index
                .astype(str)
            )

            chemical_similarity.columns = (
                chemical_similarity
                .columns
                .astype(str)
            )

            chemical_similarity = (
                chemical_similarity.apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            print(
                f"  ✓ Chemical matrix: "
                f"{chemical_similarity.shape}"
            )

        except Exception as e:

            print(
                f"  ⚠ Chemical matrix error: {e}"
            )

    return {

        "hepg2_profiles":
            hepg2_profiles,

        "hepg2_coords":
            hepg2_coords,

        "drug_annotations":
            drug_annotations,

        "nes_matrix":  # 🆕 Returns the NES matrix instead of dynamic pathway features
            nes_matrix,

        "chemical_similarity":
            chemical_similarity
    }


# =============================================================================
# Query Samples
# =============================================================================

def load_query_samples():
    print("\n📂 Loading query samples...")

    query_samples = {}

    if not os.path.exists(
            PROJECTION_RESULTS_ROOT
    ):
        print(
            f"❌ Path does not exist: "
            f"{PROJECTION_RESULTS_ROOT}"
        )

        return query_samples

    drug_folders = [
        d
        for d in os.listdir(
            PROJECTION_RESULTS_ROOT
        )
        if os.path.isdir(
            os.path.join(
                PROJECTION_RESULTS_ROOT,
                d
            )
        )
    ]

    # -------------------------------------------------------------------------
    # For testing, you can change to drug_folders[:5]
    # For formal analysis, use drug_folders
    # -------------------------------------------------------------------------

    for drug_name in tqdm(
            drug_folders,
            desc="Loading queries"
    ):

        try:

            coords_file = os.path.join(
                PROJECTION_RESULTS_ROOT,
                drug_name,
                "projection_coordinates.csv"
            )

            if not os.path.exists(coords_file):
                continue

            projection_coords = pd.read_csv(
                coords_file
            )

            drug_folder = os.path.join(
                DRUG_PROFILES_ROOT,
                drug_name
            )

            if not os.path.isdir(drug_folder):
                continue

            cellline_data = {}

            csv_files = [
                f
                for f in os.listdir(
                    drug_folder
                )
                if (
                        f.endswith(".csv")
                        and not f.endswith(
                    "_extraction_info.csv"
                )
                )
            ]

            for csv_file in csv_files:

                cellline_name = (
                    csv_file.replace(
                        ".csv",
                        ""
                    )
                )

                file_path = os.path.join(
                    drug_folder,
                    csv_file
                )

                try:

                    data = pd.read_csv(
                        file_path,
                        index_col=0
                    )

                    # conditions × proteins
                    cellline_data[
                        cellline_name
                    ] = data.T

                except Exception:
                    continue

            if len(cellline_data) > 0:
                query_samples[
                    str(drug_name)
                ] = {

                    "projection_coords":
                        projection_coords,

                    "cellline_profiles":
                        cellline_data
                }

        except Exception:
            continue

    print(
        f"  ✓ Query drugs: "
        f"{len(query_samples)}"
    )

    return query_samples


# =============================================================================
# MOA Ground Truth
# =============================================================================

def get_moa_ground_truth(
        query_drug,
        drug_annotations,
        hepg2_coords,
        moa_dimensions
):
    """
    Ground truth also strictly excludes the query identity.
    """

    query_drug = str(query_drug)

    ground_truths = {}

    reference_drugs = set(
        hepg2_coords["Drug"].astype(str)
    )

    # -------------------------------------------------------------------------
    # DRP
    # -------------------------------------------------------------------------

    if "DRP" in moa_dimensions:

        query_info = hepg2_coords[
            hepg2_coords["Drug"]
            == query_drug
            ]

        if len(query_info) > 0:
            query_drp = (
                query_info.iloc[0]["DRP"]
            )

            same_drp = set(
                hepg2_coords[
                    (
                            hepg2_coords["DRP"]
                            == query_drp
                    )
                    &
                    (
                            hepg2_coords["Drug"]
                            != query_drug
                    )
                    ]["Drug"].astype(str)
            )

            ground_truths["DRP"] = (
                    same_drp
                    & reference_drugs
            )

    # -------------------------------------------------------------------------
    # Annotation MOA
    # -------------------------------------------------------------------------

    if (
            drug_annotations is not None
            and query_drug
            in drug_annotations.index
    ):

        query_annotation = (
            drug_annotations.loc[
                query_drug
            ]
        )

        # If duplicate index exists, take the first row
        if isinstance(
                query_annotation,
                pd.DataFrame
        ):
            query_annotation = (
                query_annotation.iloc[0]
            )

        for dim in moa_dimensions:

            if dim == "DRP":
                continue

            if dim not in (
                    drug_annotations.columns
            ):
                continue

            query_value = (
                query_annotation[dim]
            )

            if (
                    pd.isna(query_value)
                    or str(query_value).strip()
                    in [
                "",
                "Unknown",
                "nan"
            ]
            ):
                ground_truths[dim] = set()
                continue

            if dim == "Pathways":

                query_pathways = set(
                    p.strip()
                    for p in str(
                        query_value
                    ).split(";")
                    if p.strip()
                )

                similar_drugs = []

                for candidate, row \
                        in drug_annotations.iterrows():

                    candidate = str(candidate)

                    # Identity-Out
                    if candidate == query_drug:
                        continue

                    if candidate not in reference_drugs:
                        continue

                    value = row[dim]

                    if pd.isna(value):
                        continue

                    candidate_pathways = set(
                        p.strip()
                        for p in str(
                            value
                        ).split(";")
                        if p.strip()
                    )

                    if (
                            query_pathways
                            & candidate_pathways
                    ):
                        similar_drugs.append(
                            candidate
                        )

                ground_truths[dim] = set(
                    similar_drugs
                )

            else:

                matching = []

                for candidate, row \
                        in drug_annotations.iterrows():

                    candidate = str(candidate)

                    # Identity-Out
                    if candidate == query_drug:
                        continue

                    if candidate not in reference_drugs:
                        continue

                    candidate_value = (
                        row[dim]
                    )

                    if pd.isna(
                            candidate_value
                    ):
                        continue

                    if (
                            candidate_value
                            == query_value
                    ):
                        matching.append(
                            candidate
                        )

                ground_truths[dim] = set(
                    matching
                )

    # Final safeguard
    for dim in ground_truths:
        ground_truths[dim].discard(
            query_drug
        )

    return ground_truths


# =============================================================================
# Representation Base
# =============================================================================

class DrugRepresentation:

    def __init__(self, name):
        self.name = name

    def find_similar_drugs(
            self,
            query_input,
            query_drug,
            reference_data,
            top_k=10
    ):
        raise NotImplementedError


# =============================================================================
# 1. UMAP Identity-Out
# =============================================================================

class UMAPProjectionRepresentation(
    DrugRepresentation
):

    def __init__(self):

        super().__init__(
            "UMAP_Projection"
        )

    def find_similar_drugs(
            self,
            query_coords,
            query_drug,
            reference_data,
            top_k=10
    ):

        try:

            query_u1, query_u2 = (
                query_coords
            )

            ref = (
                reference_data[
                    "hepg2_coords"
                ]
            )

            distances = []

            for _, row in ref.iterrows():

                candidate = str(
                    row["Drug"]
                )

                # =============================================================
                # STRICT IDENTITY-OUT
                # =============================================================

                if candidate == query_drug:
                    continue

                distance = np.sqrt(
                    (
                            float(row["UMAP1"])
                            - float(query_u1)
                    ) ** 2
                    +
                    (
                            float(row["UMAP2"])
                            - float(query_u2)
                    ) ** 2
                )

                distances.append(
                    (
                        candidate,
                        distance
                    )
                )

            distances.sort(
                key=lambda x: x[1]
            )

            result = [
                drug
                for drug, _
                in distances[:top_k]
            ]

            assert_identity_out(
                result,
                query_drug,
                self.name
            )

            return result

        except Exception as e:

            print(
                f"UMAP failed: {e}"
            )

            return []


# =============================================================================
# 2. Chemical Fingerprint Identity-Out
# =============================================================================

class ChemicalFingerprintRepresentation(
    DrugRepresentation
):

    def __init__(self):

        super().__init__(
            "Chemical_Fingerprint"
        )

    def find_similar_drugs(
            self,
            query_input,
            query_drug,
            reference_data,
            top_k=10
    ):

        try:

            similarity_matrix = (
                reference_data[
                    "chemical_similarity"
                ]
            )

            if similarity_matrix is None:
                return []

            if (
                    query_drug
                    not in similarity_matrix.index
            ):
                return []

            similarities = (
                similarity_matrix
                .loc[query_drug]
                .copy()
            )

            # =============================================================
            # STRICT IDENTITY-OUT
            # =============================================================

            similarities = similarities[
                similarities.index
                != query_drug
                ]

            # Only retain HepG2 candidates
            reference_drugs = set(
                reference_data[
                    "hepg2_coords"
                ]["Drug"].astype(str)
            )

            similarities = similarities[
                [
                    idx in reference_drugs
                    for idx
                    in similarities.index
                ]
            ]

            similarities = (
                similarities
                .dropna()
                .sort_values(
                    ascending=False
                )
            )

            result = (
                similarities
                .head(top_k)
                .index
                .astype(str)
                .tolist()
            )

            assert_identity_out(
                result,
                query_drug,
                self.name
            )

            return result

        except Exception as e:

            print(
                f"Chemical failed: {e}"
            )

            return []


# =============================================================================
# 🆕 3. NES Matrix Pathway Identity-Out
# =============================================================================

class PathwayNESMatrixRepresentation(DrugRepresentation):
    """🆕 Pathway representation method directly using pre-computed NES matrix"""

    def __init__(self, similarity_metric="spearman"):
        super().__init__(f"Pathway_NES_Matrix_{similarity_metric}")
        self.similarity_metric = similarity_metric

    def find_similar_drugs(
            self,
            query_profile,  # This parameter is not actually needed when using pre-computed matrices
            query_drug,
            reference_data,
            top_k=10
    ):

        try:
            # Directly use the pre-loaded NES matrix
            nes_matrix = reference_data["nes_matrix"]

            if nes_matrix is None:
                print("NES matrix not available")
                return []

            # Check if the query drug is in the NES matrix
            if query_drug not in nes_matrix.index:
                print(f"Query drug {query_drug} not found in NES matrix")
                return []

            query_nes = nes_matrix.loc[query_drug]
            similarities = {}

            for candidate in nes_matrix.index:
                candidate = str(candidate)

                # =============================================================
                # STRICT IDENTITY-OUT
                # =============================================================
                if candidate == query_drug:
                    continue

                ref_nes = nes_matrix.loc[candidate]

                # Calculate similarity
                similarity = self._calculate_nes_similarity(
                    query_nes,
                    ref_nes,
                    self.similarity_metric
                )

                similarities[candidate] = similarity

            # Sort and return Top-K
            sorted_drugs = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            result = [drug for drug, _ in sorted_drugs[:top_k]]

            assert_identity_out(result, query_drug, self.name)

            return result

        except Exception as e:
            print(f"Pathway NES Matrix failed: {e}")
            traceback.print_exc()
            return []

    def _calculate_nes_similarity(
            self,
            query_nes,
            reference_nes,
            similarity_metric
    ):
        """Calculate similarity between two NES vectors"""

        # Convert to numeric and handle missing values
        q = pd.to_numeric(query_nes, errors="coerce").fillna(0)
        r = pd.to_numeric(reference_nes, errors="coerce").fillna(0)

        try:
            if similarity_metric == "spearman":
                similarity, _ = spearmanr(q, r)
            elif similarity_metric == "pearson":
                similarity, _ = pearsonr(q, r)
            elif similarity_metric == "cosine":
                similarity = cosine_similarity([q.values], [r.values])[0][0]
            else:
                # Euclidean distance
                distance = euclidean(q.values, r.values)
                similarity = 1.0 / (1.0 + distance)

            return float(similarity) if not np.isnan(similarity) else 0.0

        except Exception:
            return 0.0


# =============================================================================
# 4/5. Protein Identity-Out
# =============================================================================

class ProteinDistanceRepresentation(
    DrugRepresentation
):

    def __init__(
            self,
            distance_metric="cosine"
    ):

        super().__init__(
            f"Protein_{distance_metric.title()}"
        )

        self.distance_metric = (
            distance_metric
        )

    def find_similar_drugs(
            self,
            query_profile,
            query_drug,
            reference_data,
            top_k=10
    ):

        try:

            hepg2_profiles = (
                reference_data[
                    "hepg2_profiles"
                ]
            )

            # =============================================================
            # STRICT IDENTITY-OUT
            #
            # The query drug is removed before standardization.
            # This is stricter than removing it after calculations.
            # =============================================================

            identity_out_profiles = (
                hepg2_profiles[
                    hepg2_profiles.index
                    != query_drug
                    ]
            )

            if isinstance(
                    query_profile,
                    pd.Series
            ):
                query_profile = (
                    query_profile
                    .to_frame()
                    .T
                )

            ref_proteins = set(
                identity_out_profiles.columns
            )

            query_proteins = set(
                query_profile.columns
            )

            common_proteins = sorted(
                ref_proteins
                & query_proteins
            )

            if len(common_proteins) < 500:
                return []

            ref_aligned = (
                identity_out_profiles[
                    common_proteins
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            query_aligned = (
                query_profile[
                    common_proteins
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            # Median imputation for missing values
            ref_medians = (
                ref_aligned.median(axis=0)
            )

            ref_aligned = (
                ref_aligned.fillna(
                    ref_medians
                )
            )

            query_aligned = (
                query_aligned.fillna(
                    ref_medians
                )
            )

            # Drop features that are still unusable
            valid_cols = [
                c
                for c in common_proteins
                if (
                        np.isfinite(
                            ref_aligned[c]
                            .values
                        ).all()
                        and
                        np.isfinite(
                            query_aligned[c]
                            .values
                        ).all()
                )
            ]

            if len(valid_cols) < 500:
                return []

            ref_aligned = (
                ref_aligned[
                    valid_cols
                ]
            )

            query_aligned = (
                query_aligned[
                    valid_cols
                ]
            )

            # =============================================================
            # Scaler only fits the identity-out reference
            # =============================================================

            scaler = StandardScaler()

            ref_std = scaler.fit_transform(
                ref_aligned.values
            )

            query_std = scaler.transform(
                query_aligned.values
            )

            query_vec = (
                query_std[0]
            )

            similarities = {}

            if self.distance_metric == "cosine":

                values = cosine_similarity(
                    [query_vec],
                    ref_std
                )[0]

                similarities = dict(
                    zip(
                        ref_aligned.index.astype(str),
                        values
                    )
                )

            elif (
                    self.distance_metric
                    == "euclidean"
            ):

                distances = np.linalg.norm(
                    ref_std - query_vec,
                    axis=1
                )

                values = (
                        1.0
                        / (1.0 + distances)
                )

                similarities = dict(
                    zip(
                        ref_aligned.index.astype(str),
                        values
                    )
                )

            elif (
                    self.distance_metric
                    == "spearman"
            ):

                for j, candidate \
                        in enumerate(
                    ref_aligned.index
                ):

                    corr, _ = spearmanr(
                        query_vec,
                        ref_std[j]
                    )

                    if np.isnan(corr):
                        corr = 0.0

                    similarities[
                        str(candidate)
                    ] = float(corr)

            sorted_drugs = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            result = [
                drug
                for drug, _
                in sorted_drugs[:top_k]
            ]

            assert_identity_out(
                result,
                query_drug,
                self.name
            )

            return result

        except Exception as e:

            print(
                f"Protein failed: {e}"
            )

            return []


# =============================================================================
# 6. Random Identity-Out
# =============================================================================

class RandomBaseline(
    DrugRepresentation
):

    def __init__(
            self,
            seed=1000
    ):

        super().__init__(
            "Random_Baseline"
        )

        self.seed = seed

    def _stable_seed(
            self,
            query_drug
    ):

        text = (
            f"{self.seed}_{query_drug}"
        )

        digest = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        return int(
            digest[:8],
            16
        )

    def find_similar_drugs(
            self,
            query_input,
            query_drug,
            reference_data,
            top_k=10
    ):

        try:

            available_drugs = (
                reference_data[
                    "hepg2_coords"
                ]["Drug"]
                .astype(str)
                .unique()
            )

            # =============================================================
            # STRICT IDENTITY-OUT
            # =============================================================

            available_drugs = np.array(
                [
                    d
                    for d in available_drugs
                    if d != query_drug
                ]
            )

            rng = np.random.default_rng(
                self._stable_seed(
                    query_drug
                )
            )

            if (
                    len(available_drugs)
                    <= top_k
            ):

                result = (
                    available_drugs.tolist()
                )

            else:

                result = (
                    rng.choice(
                        available_drugs,
                        size=top_k,
                        replace=False
                    )
                    .tolist()
                )

            assert_identity_out(
                result,
                query_drug,
                self.name
            )

            return result

        except Exception as e:

            print(
                f"Random failed: {e}"
            )

            return []


# =============================================================================
# Retrieval Metrics
# =============================================================================

def calculate_retrieval_metrics(
        retrieved_drugs,
        ground_truth,
        k_values=None
):
    if k_values is None:
        k_values = K_VALUES

    metrics = {}

    if len(ground_truth) == 0:

        for k in k_values:
            metrics[
                f"Precision@{k}"
            ] = np.nan

            metrics[
                f"Recall@{k}"
            ] = np.nan

            metrics[
                f"Hit@{k}"
            ] = np.nan

        metrics["MRR"] = np.nan
        metrics["MAP"] = np.nan

        return metrics

    for k in k_values:

        top_k = (
            retrieved_drugs[:k]
        )

        if len(top_k) == 0:

            precision = 0.0
            recall = 0.0
            hit = 0.0

        else:

            hits = len(
                set(top_k)
                & ground_truth
            )

            precision = (
                    hits / len(top_k)
            )

            recall = (
                    hits / len(
                ground_truth
            )
            )

            hit = (
                1.0
                if hits > 0
                else 0.0
            )

        metrics[
            f"Precision@{k}"
        ] = precision

        metrics[
            f"Recall@{k}"
        ] = recall

        metrics[
            f"Hit@{k}"
        ] = hit

    # MRR
    mrr = 0.0

    for rank, drug in enumerate(
            retrieved_drugs,
            start=1
    ):

        if drug in ground_truth:
            mrr = 1.0 / rank
            break

    metrics["MRR"] = mrr

    # Retained naming as MAP based on original script
    precisions = []
    num_hits = 0

    for rank, drug in enumerate(
            retrieved_drugs,
            start=1
    ):

        if drug in ground_truth:
            num_hits += 1

            precisions.append(
                num_hits / rank
            )

    metrics["MAP"] = (
        np.mean(precisions)
        if precisions
        else 0.0
    )

    return metrics


# =============================================================================
# 🆕 Main Comparison (using NES matrix)
# =============================================================================

def compare_drug_representations_identity_out(
        query_samples,
        reference_data,
        logger=None
):
    print("\n" + "=" * 80)
    print(
        "STRICT IDENTITY-OUT "
        "DRUG REPRESENTATION BENCHMARK"
    )
    print("Using Pre-computed NES Matrix for Pathway Analysis")
    print("=" * 80)

    representations = {

        "UMAP_Projection":
            UMAPProjectionRepresentation(),

        "Chemical_Fingerprint":
            ChemicalFingerprintRepresentation(),

        "Pathway_NES_Matrix":  # 🆕 Version using NES matrix
            PathwayNESMatrixRepresentation(
                "spearman"
            ),

        "Protein_Cosine":
            ProteinDistanceRepresentation(
                "cosine"
            ),

        "Protein_Euclidean":
            ProteinDistanceRepresentation(
                "euclidean"
            ),

        "Random_Baseline":
            RandomBaseline(
                RANDOM_SEED
            )
    }

    print("\nMethods:")

    for name in representations:
        print(
            f"  ✓ {name}: identity-out"
        )

    # 🆕 Check NES matrix availability
    if reference_data["nes_matrix"] is None:
        print("  ⚠️  NES matrix unavailable, skipping Pathway method")
        del representations["Pathway_NES_Matrix"]

    all_results = []
    failed_cases = []
    retrieval_records = []

    total_samples = sum(
        len(
            sample_data[
                "projection_coords"
            ]
        )
        for sample_data
        in query_samples.values()
    )

    progress = tqdm(
        total=total_samples,
        desc="Identity-out benchmark"
    )

    reference_drugs = set(
        reference_data[
            "hepg2_coords"
        ]["Drug"].astype(str)
    )

    for drug_name, sample_data \
            in query_samples.items():

        drug_name = str(drug_name)

        # Query must have identity information for identity-out benchmark
        if drug_name not in reference_drugs:

            if logger:
                logger.warning(
                    f"{drug_name} not in HepG2 reference, "
                    f"skipping identity-out evaluation"
                )

            progress.update(
                len(
                    sample_data[
                        "projection_coords"
                    ]
                )
            )

            continue

        ground_truths = (
            get_moa_ground_truth(
                query_drug=drug_name,
                drug_annotations=(
                    reference_data[
                        "drug_annotations"
                    ]
                ),
                hepg2_coords=(
                    reference_data[
                        "hepg2_coords"
                    ]
                ),
                moa_dimensions=(
                    MOA_DIMENSIONS
                )
            )
        )

        valid_gt = {

            dim: gt
            for dim, gt
            in ground_truths.items()
            if len(gt) > 0
        }

        if len(valid_gt) == 0:
            progress.update(
                len(
                    sample_data[
                        "projection_coords"
                    ]
                )
            )

            continue

        projection_coords = (
            sample_data[
                "projection_coords"
            ]
        )

        cellline_profiles = (
            sample_data[
                "cellline_profiles"
            ]
        )

        for _, coord_row \
                in projection_coords.iterrows():

            try:

                cellline = str(
                    coord_row["Cellline"]
                )

                condition = (
                    coord_row["Condition"]
                )

                projected_u1 = float(
                    coord_row[
                        "UMAP1_Projected"
                    ]
                )

                projected_u2 = float(
                    coord_row[
                        "UMAP2_Projected"
                    ]
                )

                if (
                        cellline
                        not in cellline_profiles
                ):
                    progress.update(1)
                    continue

                profile_df = (
                    cellline_profiles[
                        cellline
                    ]
                )

                # Index types might differ
                if (
                        condition
                        not in profile_df.index
                ):

                    condition_str = str(
                        condition
                    )

                    index_as_str = (
                        profile_df.index
                        .astype(str)
                    )

                    matches = (
                            index_as_str
                            == condition_str
                    )

                    if not matches.any():
                        progress.update(1)
                        continue

                    condition_position = (
                        np.where(matches)[0][0]
                    )

                    condition_profile = (
                        profile_df.iloc[
                            condition_position
                        ]
                    )

                else:

                    condition_profile = (
                        profile_df.loc[
                            condition
                        ]
                    )

                representation_results = {}

                # -------------------------------------------------------------
                # UMAP
                # -------------------------------------------------------------

                representation_results[
                    "UMAP_Projection"
                ] = (
                    representations[
                        "UMAP_Projection"
                    ]
                    .find_similar_drugs(
                        query_coords=(
                            projected_u1,
                            projected_u2
                        ),
                        query_drug=drug_name,
                        reference_data=(
                            reference_data
                        ),
                        top_k=max(
                            K_VALUES
                        )
                    )
                )

                # -------------------------------------------------------------
                # Chemical
                # -------------------------------------------------------------

                representation_results[
                    "Chemical_Fingerprint"
                ] = (
                    representations[
                        "Chemical_Fingerprint"
                    ]
                    .find_similar_drugs(
                        query_input=None,
                        query_drug=drug_name,
                        reference_data=(
                            reference_data
                        ),
                        top_k=max(
                            K_VALUES
                        )
                    )
                )

                # -------------------------------------------------------------
                # 🆕 Pathway NES Matrix
                # -------------------------------------------------------------

                if "Pathway_NES_Matrix" in representations:
                    representation_results[
                        "Pathway_NES_Matrix"
                    ] = (
                        representations[
                            "Pathway_NES_Matrix"
                        ]
                        .find_similar_drugs(
                            query_profile=None,  # NES method does not need external profiles
                            query_drug=drug_name,
                            reference_data=(
                                reference_data
                            ),
                            top_k=max(
                                K_VALUES
                            )
                        )
                    )

                # -------------------------------------------------------------
                # Protein cosine
                # -------------------------------------------------------------

                representation_results[
                    "Protein_Cosine"
                ] = (
                    representations[
                        "Protein_Cosine"
                    ]
                    .find_similar_drugs(
                        query_profile=(
                            condition_profile
                        ),
                        query_drug=drug_name,
                        reference_data=(
                            reference_data
                        ),
                        top_k=max(
                            K_VALUES
                        )
                    )
                )

                # -------------------------------------------------------------
                # Protein Euclidean
                # -------------------------------------------------------------

                representation_results[
                    "Protein_Euclidean"
                ] = (
                    representations[
                        "Protein_Euclidean"
                    ]
                    .find_similar_drugs(
                        query_profile=(
                            condition_profile
                        ),
                        query_drug=drug_name,
                        reference_data=(
                            reference_data
                        ),
                        top_k=max(
                            K_VALUES
                        )
                    )
                )

                # -------------------------------------------------------------
                # Random
                # -------------------------------------------------------------

                representation_results[
                    "Random_Baseline"
                ] = (
                    representations[
                        "Random_Baseline"
                    ]
                    .find_similar_drugs(
                        query_input=None,
                        query_drug=drug_name,
                        reference_data=(
                            reference_data
                        ),
                        top_k=max(
                            K_VALUES
                        )
                    )
                )

                # =============================================================
                # GLOBAL IDENTITY LEAKAGE CHECK
                # =============================================================

                for method_name, retrieved \
                        in representation_results.items():
                    assert_identity_out(
                        retrieved,
                        drug_name,
                        method_name
                    )

                    retrieval_records.append({

                        "Query_Drug":
                            drug_name,

                        "Cellline":
                            cellline,

                        "Condition":
                            condition,

                        "Representation_Method":
                            method_name,

                        "Identity_Excluded":
                            True,

                        "Top10":
                            ";".join(
                                retrieved
                            )
                    })

                # -------------------------------------------------------------
                # Evaluation
                # -------------------------------------------------------------

                for moa_dimension, ground_truth \
                        in valid_gt.items():

                    # insurance
                    if drug_name in ground_truth:
                        raise RuntimeError(
                            "Ground truth "
                            "identity leakage: "
                            f"{drug_name}"
                        )

                    for method_name, retrieved \
                            in representation_results.items():

                        if len(retrieved) == 0:
                            continue

                        metrics = (
                            calculate_retrieval_metrics(
                                retrieved,
                                ground_truth,
                                K_VALUES
                            )
                        )

                        row = {

                            "Query_Drug":
                                drug_name,

                            "Cellline":
                                cellline,

                            "Condition":
                                condition,

                            "Representation_Method":
                                method_name,

                            "MOA_Dimension":
                                moa_dimension,

                            "Identity_Out":
                                True,

                            "Query_In_Retrieved":
                                (
                                        drug_name
                                        in retrieved
                                ),

                            "Ground_Truth_Size":
                                len(
                                    ground_truth
                                ),

                            **metrics
                        }

                        all_results.append(
                            row
                        )

                progress.update(1)

            except Exception as e:

                failed_cases.append({

                    "Query_Drug":
                        drug_name,

                    "Cellline":
                        coord_row.get(
                            "Cellline",
                            ""
                        ),

                    "Condition":
                        coord_row.get(
                            "Condition",
                            ""
                        ),

                    "Error":
                        str(e),

                    "Traceback":
                        traceback.format_exc()
                })

                if logger:
                    logger.error(
                        f"Failed: "
                        f"{drug_name} - {e}"
                    )

                progress.update(1)

    progress.close()

    results_df = pd.DataFrame(
        all_results
    )

    retrieval_df = pd.DataFrame(
        retrieval_records
    )

    if len(retrieval_df) > 0:
        retrieval_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "identity_out_retrieval_lists.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    if failed_cases:
        pd.DataFrame(
            failed_cases
        ).to_csv(
            os.path.join(
                OUTPUT_DIR,
                "failed_identity_out_cases.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    return results_df


# =============================================================================
# Validate Identity-Out
# =============================================================================

def validate_identity_out_results(
        results_df
):
    print("\n🔍 Identity-Out validation")
    print("-" * 60)

    if len(results_df) == 0:
        print(
            "No results available."
        )

        return False

    leakage = (
        results_df[
            "Query_In_Retrieved"
        ].sum()
    )

    print(
        f"Total evaluation rows: "
        f"{len(results_df)}"
    )

    print(
        f"Identity leakage rows: "
        f"{leakage}"
    )

    if leakage == 0:
        print(
            "✅ STRICT IDENTITY-OUT PASSED"
        )

        return True

    print(
        "❌ IDENTITY LEAKAGE DETECTED"
    )

    return False


# =============================================================================
# Summary
# =============================================================================

def create_summary(
        results_df
):
    if len(results_df) == 0:
        return pd.DataFrame()

    metrics = [
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "Precision@10",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Hit@10",
        "MRR",
        "MAP"
    ]

    summary = (
        results_df
        .groupby(
            "Representation_Method"
        )[metrics]
        .mean()
        .reset_index()
    )

    score_columns = [
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "Precision@10",
        "MRR",
        "MAP"
    ]

    summary[
        "Overall_Score"
    ] = (
        summary[
            score_columns
        ]
        .mean(
            axis=1,
            skipna=True
        )
    )

    summary = (
        summary
        .sort_values(
            "Overall_Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    summary[
        "Rank"
    ] = (
        np.arange(
            1,
            len(summary) + 1
        )
    )

    return summary


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(
        results_df,
        summary_df
):
    if len(results_df) == 0:
        return

    colors = {

        "UMAP_Projection":
            "#FF6B6B",

        "Chemical_Fingerprint":
            "#4ECDC4",

        "Pathway_NES_Matrix":  # 🆕 Update color
            "#FF9F43",

        "Protein_Cosine":
            "#96CEB4",

        "Protein_Euclidean":
            "#FECA57",

        "Random_Baseline":
            "#CCCCCC"
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )

    # -------------------------------------------------------------------------
    # Precision@5 heatmap
    # -------------------------------------------------------------------------

    pivot = results_df.pivot_table(
        values="Precision@5",
        index="Representation_Method",
        columns="MOA_Dimension",
        aggfunc="mean"
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title(
        "Identity-Out Precision@5\n(Using NES Matrix)"
    )

    # -------------------------------------------------------------------------
    # MRR heatmap
    # -------------------------------------------------------------------------

    pivot_mrr = (
        results_df.pivot_table(
            values="MRR",
            index="Representation_Method",
            columns="MOA_Dimension",
            aggfunc="mean"
        )
    )

    sns.heatmap(
        pivot_mrr,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=axes[0, 1]
    )

    axes[0, 1].set_title(
        "Identity-Out MRR\n(Using NES Matrix)"
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    ranking = (
        summary_df.sort_values(
            "Overall_Score",
            ascending=True
        )
    )

    ranking_colors = [
        colors.get(
            method,
            "#999999"
        )
        for method
        in ranking[
            "Representation_Method"
        ]
    ]

    axes[1, 0].barh(
        ranking[
            "Representation_Method"
        ],
        ranking[
            "Overall_Score"
        ],
        color=ranking_colors
    )

    axes[1, 0].set_title(
        "Overall Identity-Out Ranking\n(NES Matrix Based)"
    )

    axes[1, 0].set_xlabel(
        "Overall Score"
    )

    # -------------------------------------------------------------------------
    # Precision K
    # -------------------------------------------------------------------------

    for method in (
            results_df[
                "Representation_Method"
            ].unique()
    ):
        subset = results_df[
            results_df[
                "Representation_Method"
            ] == method
            ]

        values = [
            subset[
                f"Precision@{k}"
            ].mean()
            for k in K_VALUES
        ]

        axes[1, 1].plot(
            K_VALUES,
            values,
            marker="o",
            linewidth=2,
            label=method,
            color=colors.get(
                method,
                "#999999"
            )
        )

    axes[1, 1].set_xlabel("K")
    axes[1, 1].set_ylabel("Precision@K")

    axes[1, 1].set_title(
        "Identity-Out Precision@K\n(NES Matrix Based)"
    )

    axes[1, 1].legend(
        fontsize=8
    )

    axes[1, 1].grid(
        alpha=0.3
    )

    plt.suptitle(
        "Drug Representation Benchmark V4\n"
        "Strict Drug Identity-Out Retrieval with Pre-computed NES Matrix",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "identity_out_comparison_nes.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "identity_out_comparison_nes.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 🆕 Main (Using NES Matrix)
# =============================================================================

def main():
    logger = create_logger()

    print(
        "\n"
        "============================================================\n"
        " STRICT IDENTITY-OUT DRUG REPRESENTATION BENCHMARK V4\n"
        " Using Pre-computed NES Matrix for Pathway Analysis\n"
        "============================================================"
    )

    start = datetime.now()

    logger.info(
        "Starting NES matrix-based strict identity-out benchmark"
    )

    try:

        # ---------------------------------------------------------------------
        # 1. Reference
        # ---------------------------------------------------------------------

        reference_data = (
            load_reference_data()
        )

        # ---------------------------------------------------------------------
        # 🆕 2. Diagnose NES Matrix
        # ---------------------------------------------------------------------

        nes_available = diagnose_nes_matrix(
            reference_data["nes_matrix"],
            reference_data["hepg2_coords"]
        )

        if not nes_available:
            print("⚠️  NES matrix quality issues; Pathway method may perform poorly")

        # ---------------------------------------------------------------------
        # 3. Queries
        # ---------------------------------------------------------------------

        query_samples = (
            load_query_samples()
        )

        if len(query_samples) == 0:
            logger.error(
                "No query samples."
            )

            return

        # ---------------------------------------------------------------------
        # 4. Benchmark
        # ---------------------------------------------------------------------

        results_df = (
            compare_drug_representations_identity_out(
                query_samples,
                reference_data,
                logger
            )
        )

        if len(results_df) == 0:
            logger.error(
                "No evaluation results."
            )

            return

        # ---------------------------------------------------------------------
        # 5. Strict leakage validation
        # ---------------------------------------------------------------------

        passed = (
            validate_identity_out_results(
                results_df
            )
        )

        if not passed:
            raise RuntimeError(
                "Identity-out validation failed."
            )

        # ---------------------------------------------------------------------
        # 6. Save raw results
        # ---------------------------------------------------------------------

        results_path = os.path.join(
            OUTPUT_DIR,
            "identity_out_results_nes.csv"
        )

        results_df.to_csv(
            results_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 7. Summary
        # ---------------------------------------------------------------------

        summary_df = (
            create_summary(
                results_df
            )
        )

        summary_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "identity_out_ranking_nes.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        # ---------------------------------------------------------------------
        # 8. MOA summaries
        # ---------------------------------------------------------------------

        metrics = [
            "Precision@1",
            "Precision@3",
            "Precision@5",
            "Precision@10",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR",
            "MAP"
        ]

        for metric in metrics:
            pivot = (
                results_df.pivot_table(
                    values=metric,
                    index=(
                        "Representation_Method"
                    ),
                    columns="MOA_Dimension",
                    aggfunc="mean"
                )
            )

            pivot.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"identity_out_nes_{metric}.csv"
                ),
                encoding="utf-8-sig"
            )

        # ---------------------------------------------------------------------
        # 9. Visualization
        # ---------------------------------------------------------------------

        create_visualization(
            results_df,
            summary_df
        )

        # ---------------------------------------------------------------------
        # 10. Print results
        # ---------------------------------------------------------------------

        duration = (
                datetime.now()
                - start
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            "✅ NES MATRIX IDENTITY-OUT BENCHMARK COMPLETE"
        )

        print(
            "============================================================"
        )

        print(
            f"\nEvaluation rows: "
            f"{len(results_df)}"
        )

        print(
            f"Query drugs: "
            f"{results_df['Query_Drug'].nunique()}"
        )

        print(
            f"Identity leakage: "
            f"{results_df['Query_In_Retrieved'].sum()}"
        )

        print(
            f"Runtime: {duration}"
        )

        # 🆕 Special report for NES matrix
        if reference_data["nes_matrix"] is not None:
            nes_drugs = len(reference_data["nes_matrix"])
            nes_pathways = reference_data["nes_matrix"].shape[1]
            print(f"NES matrix: {nes_drugs} drugs × {nes_pathways} pathways")

        print(
            "\nOverall ranking:"
        )

        for _, row \
                in summary_df.iterrows():
            print(
                f"  #{int(row['Rank'])} "
                f"{row['Representation_Method']:25s} "
                f"{row['Overall_Score']:.4f}"
            )

        print(
            f"\nOutput directory:\n"
            f"{OUTPUT_DIR}"
        )

        logger.info(
            "NES matrix identity-out benchmark completed."
        )

    except Exception as e:

        logger.error(
            str(e)
        )

        logger.error(
            traceback.format_exc()
        )

        print(
            f"\n❌ Error: {e}"
        )

        traceback.print_exc()


if __name__ == "__main__":
    main()
