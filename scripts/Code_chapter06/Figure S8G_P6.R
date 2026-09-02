script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
script_dir <- if (length(file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = FALSE))
} else {
  getwd()
}
repo_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
input_dir <- file.path(repo_root, "data", "Code_chapter06")
chapter04_results_dir <- file.path(repo_root, "results", "Code_chapter04")

library(openxlsx)

# Load external proteomics data (rows = proteins, columns = compounds)
external_l2fc <- read.csv(
  file.path(input_dir, "TCM_24_combat_matrix.csv"),
  check.names = FALSE,
  header = TRUE,
  row.names = 1
)

# Load external drug-response labels
external_labels <- read.csv(
  file.path(input_dir, "hepg2_drug_screening_results_20260708_093606_mannul.csv")
)

cat(sprintf(
  "External dataset: %d compounds, %d proteins\n",
  ncol(external_l2fc),
  nrow(external_l2fc)
))


# 2. Check whether all CPS-6 core proteins are present in the external dataset
cps6_proteins <- c(
  "O60907",
  "Q9NVJ2",
  "P54920",
  "Q9GZT6",
  "P33121",
  "Q6IQ23"
)

cps6_genes <- c(
  "TBL1X",
  "ARL8B",
  "NAPA",
  "CCDC90B",
  "ACSL1",
  "PLEKHA7"
)

missing_proteins <- setdiff(
  cps6_proteins,
  rownames(external_l2fc)
)

if (length(missing_proteins) > 0) {
  cat("⚠️ WARNING: The following CPS-6 proteins are missing from the external dataset:\n")
  print(missing_proteins)
  stop("Please ensure that the external dataset contains all CPS-6 proteins")
} else {
  cat("✓ All CPS-6 proteins are present in the external dataset\n")
}


# Check compound-name matching
external_compounds <- colnames(external_l2fc)
labeled_compounds <- external_labels$Compound

if (!all(labeled_compounds %in% external_compounds)) {
  cat("⚠️ Some labeled compounds were not found in the proteomics dataset\n")
}


# 3. Calculate prediction scores using the CPS-6 model

# Load coefficients from the previously trained model
cps6_coefs <- read.xlsx(
  file.path(input_dir, "Plot02_CPS6_elasticnet_coefficients.xlsx")
)

cps6_weights <- setNames(
  cps6_coefs$Coefficient,
  cps6_coefs$UniProt
)

cat("\nCPS-6 model coefficients:\n")

print(data.frame(
  Gene = cps6_coefs$Gene,
  UniProt = cps6_coefs$UniProt,
  Coefficient = round(cps6_weights, 4)
))


# Calculate the CPS-6 protein component for each compound
external_scores <- rep(
  0,
  ncol(external_l2fc)
)

names(external_scores) <- colnames(external_l2fc)

for (protein in names(cps6_weights)) {
  if (protein %in% rownames(external_l2fc)) {
    external_scores <- external_scores +
      cps6_weights[protein] * external_l2fc[protein, ]
  }
}


# ===== Correction: Properly construct the data frame =====

# Method 1: Use tibble (recommended)
library(tibble)

external_data <- tibble(
  Compound = names(external_scores),
  cps6_protein = as.numeric(external_scores)  # Explicitly convert to numeric
)


# Merge drug-response labels
external_data <- external_data %>%
  left_join(
    external_labels %>%
      dplyr::select(
        Compound,
        Classification
      ),
    by = "Compound"
  )


# If concentration information is available, include the concentration term
if ("Concentration" %in% colnames(external_labels)) {
  
  cat("\n✓ Concentration information detected; applying concentration adjustment...\n")
  
  # Load concentration coefficient from the trained model
  model_params <- read.csv(
    file.path(chapter04_results_dir, "Plot02_CPS6_model_parameters.csv")
  )
  
  conc_coef <- model_params$Value[
    model_params$Parameter == "Log10_Conc_Coefficient"
  ]
  
  intercept <- model_params$Value[
    model_params$Parameter == "Intercept"
  ]
  
  cat(sprintf(
    "  Model intercept = %.4f\n",
    intercept
  ))
  
  cat(sprintf(
    "  Concentration coefficient = %.4f\n",
    conc_coef
  ))
  
  # Merge concentration information and calculate the final CPS-6 score
  external_data <- external_data %>%
    left_join(
      external_labels %>%
        dplyr::select(
          Compound,
          Concentration
        ),
      by = "Compound"
    ) %>%
    dplyr::mutate(
      Concentration = as.numeric(Concentration),  # Ensure numeric type
      Log10_Conc = log10(Concentration),
      cps6 = intercept +
        cps6_protein +
        conc_coef * Log10_Conc
    )
  
  cat("\n✓ Complete CPS-6 score calculation finished, including concentration adjustment\n")
  
} else {
  
  cat("\n⚠️ No concentration information detected; using the protein component as the CPS-6 score\n")
  
  # If concentration information is unavailable,
  # use the protein component directly as the CPS-6 score
  external_data <- external_data %>%
    dplyr::mutate(
      cps6 = cps6_protein
    )
}


# Inspect results
cat(sprintf(
  "\nSuccessfully calculated CPS-6 scores for %d compounds\n",
  nrow(external_data)
))

cat("\nPreview of the first 5 compounds:\n")

print(
  head(
    external_data %>%
      dplyr::select(
        Compound,
        cps6,
        Classification,
        everything()
      ),
    5
  )
)


# Check for missing values
na_check <- external_data %>%
  summarise(
    NA_cps6 = sum(is.na(cps6)),
    NA_classification = sum(is.na(Classification))
  )

if (na_check$NA_cps6 > 0) {
  cat(sprintf(
    "\n⚠️ WARNING: CPS-6 scores are NA for %d compounds\n",
    na_check$NA_cps6
  ))
}


# Display score distribution
cat("\nCPS-6 score summary:\n")
print(summary(external_data$cps6))


# 4. Evaluate predictive performance in labeled samples
library(pROC)

# Select samples with known labels
labeled_data <- external_data %>%
  filter(
    Classification %in% c(
      "Effective",
      "Inactive"
    )
  )

cat(sprintf(
  "\nLabeled samples: Effective = %d, Inactive = %d\n",
  sum(labeled_data$Classification == "Effective"),
  sum(labeled_data$Classification == "Inactive")
))

if (nrow(labeled_data) >= 5) {
  
  # Create binary classification labels
  labeled_data$Label <- ifelse(
    labeled_data$Classification == "Effective",
    1,
    0
  )
  
  # Calculate the ROC curve
  roc_external <- roc(
    labeled_data$Label,
    labeled_data$cps6,
    quiet = TRUE
  )
  
  ci_external <- ci.auc(
    roc_external
  )
  
  cat("\nExternal validation performance:\n")
  
  cat(sprintf(
    "  AUC = %.3f (95%% CI: %.3f - %.3f)\n",
    auc(roc_external),
    ci_external[1],
    ci_external[3]
  ))
  
  # Determine the optimal threshold
  coords_ext <- coords(
    roc_external,
    "best",
    best.method = "youden",
    ret = c(
      "threshold",
      "sensitivity",
      "specificity"
    )
  )
  
  cat(sprintf(
    "  Optimal threshold = %.3f\n",
    coords_ext$threshold
  ))
  
  cat(sprintf(
    "  Sensitivity = %.3f (%.1f%%)\n",
    coords_ext$sensitivity,
    coords_ext$sensitivity * 100
  ))
  
  cat(sprintf(
    "  Specificity = %.3f (%.1f%%)\n",
    coords_ext$specificity,
    coords_ext$specificity * 100
  ))
  
  # Wilcoxon rank-sum test
  wilcox_test <- wilcox.test(
    cps6 ~ Classification,
    data = labeled_data
  )
  
  cat(sprintf(
    "  Wilcoxon test p-value = %.2e\n",
    wilcox_test$p.value
  ))
  
} else {
  
  cat("\n⚠️ Too few labeled samples (<5); statistical testing will be skipped\n")
}


# 5. Visualize prediction results
library(ggplot2)


# 5.1 Boxplot: CPS-6 score distributions across response classes
external_data_plot <- external_data %>%
  mutate(
    Classification = factor(
      Classification,
      levels = c(
        "Effective",
        "Inactive",
        "Unknown"
      )
    )
  )

p_boxplot <- ggplot(
  external_data_plot,
  aes(
    x = Classification,
    y = cps6,
    fill = Classification
  )
) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.2,
    size = 2.5,
    alpha = 0.6
  ) +
  geom_text(
    aes(label = Compound),
    size = 2.5,
    hjust = -0.15,
    check_overlap = TRUE
  ) +
  scale_fill_manual(
    values = c(
      "Effective" = "#E89ABE",
      "Inactive" = "#74A9CF",
      "Unknown" = "#CCCCCC"
    )
  ) +
  labs(
    title = "External Validation: CPS-6 Score Distribution",
    subtitle = sprintf(
      "n = %d (Effective=%d, Inactive=%d, Unknown=%d)",
      nrow(external_data),
      sum(
        external_data$Classification == "Effective",
        na.rm = TRUE
      ),
      sum(
        external_data$Classification == "Inactive",
        na.rm = TRUE
      ),
      sum(
        external_data$Classification == "Unknown",
        na.rm = TRUE
      )
    ),
    x = "Classification",
    y = "CPS-6 Score"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(
      face = "bold"
    )
  )

print(p_boxplot)


# 5.2 ROC curve if sufficient labeled samples are available
if (exists("roc_external")) {
  
  roc_data_ext <- data.frame(
    FPR = 1 - roc_external$specificities,
    TPR = roc_external$sensitivities
  )
  
  p_roc_ext <- ggplot(
    roc_data_ext,
    aes(
      x = FPR,
      y = TPR
    )
  ) +
    geom_line(
      color = "#D73027",
      linewidth = 1.2
    ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "gray50"
    ) +
    annotate(
      "text",
      x = 0.6,
      y = 0.3,
      label = sprintf(
        "AUC = %.3f\n(%.3f - %.3f)",
        auc(roc_external),
        ci_external[1],
        ci_external[3]
      ),
      size = 5,
      fontface = "bold"
    ) +
    labs(
      title = "External Validation ROC Curve",
      subtitle = "CPS-6 Classification Performance: Effective vs Inactive",
      x = "1 - Specificity",
      y = "Sensitivity"
    ) +
    theme_bw(base_size = 12) +
    theme(
      aspect.ratio = 1
    )
  
  print(p_roc_ext)
}


# 5.3 Prediction result table
prediction_table <- external_data %>%
  arrange(desc(cps6)) %>%
  mutate(
    Rank = row_number(),
    Predicted_Class = case_when(
      cps6 > coords_ext$threshold ~ "Predicted Effective",
      cps6 <= coords_ext$threshold ~ "Predicted Inactive",
      TRUE ~ "Uncertain"
    )
  ) %>%
  dplyr::select(
    Rank,
    Compound,
    cps6,
    Classification,
    Predicted_Class
  )

print(prediction_table)
