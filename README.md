# DRP analysis workflows

This repository provides the analysis code used to generate the figures and
tables for the DRP study. It is intended for readers, collaborators, and
other users who want to inspect the workflows or reproduce the published
analyses.

The project is distributed in two parts:

- The code and documentation are maintained in Git.
- The input data and supplied analysis results are distributed as a separate
  Zenodo package because they are too large for a code-only Git repository.

> Zenodo data and results record: **add the final Zenodo DOI here**

## What the project contains

The workflows are organized by manuscript chapter:

| Chapter | Main analysis | Main entry point |
| --- | --- | --- |
| 1 | Proteomic data processing, quality control, normalization, and overview figures | `Figure 1&S1.Rmd` |
| 2 | Drug-response comparisons, variant analysis, and pathway analysis | `Figure 2&S2.Rmd` |
| 3 | HepG2/HCT116 clustering, UMAP analysis, and cluster biology | `Figure 3_P1.py`–`Figure 3_P5.py` |
| 4 | CPS-6 modeling, validation, Jurkat analysis, and L1000 analysis | `Figure 4_P1.Rmd`, `Figure 4D_P2.py` |
| 5 | Drug-pair correlations and network analysis | `Figure 5A.Rmd` |
| 6 | External projection, validation, MOA retrieval, and CPS-6 summary analyses | `Figure 6A_P2.py`–`Figure S8G_P6.R` |

## Download the code and data

Clone or download this Git repository, then download the matching `data/`
and `results/` directories from the Zenodo record. Place the two Zenodo
directories at the top level of the code repository and keep the chapter
directory names unchanged:

```text
DRP/
├── data/
│   ├── Code_chapter01/
│   ├── Code_chapter02/
│   ├── Code_chapter03/
│   ├── Code_chapter04/
│   ├── Code_chapter05/
│   └── Code_chapter06/
├── results/
│   ├── Code_chapter01/
│   ├── Code_chapter02/
│   ├── Code_chapter03/
│   ├── Code_chapter04/
│   ├── Code_chapter05/
│   └── Code_chapter06/
├── scripts/
├── requirements.yml
├── LICENSE
└── README.md
```

The `data/` directory contains the input files needed by the workflows. The
`results/` directory contains supplied outputs and also serves as the output
location when analyses are run. Do not rename files that contain spaces or
ampersands; the commands below already quote those paths correctly.

## Set up the analysis environment

The recommended environment uses Python 3.10 and R 4.3.1. From the
repository root, run:

```bash
conda env create -f requirements.yml
conda activate drp-repro
```

If the environment already exists, update it with:

```bash
conda env update -n drp-repro -f requirements.yml --prune
```

The environment file includes the Python, CRAN, and Bioconductor packages
used by the workflows, including `cmapR` for L1000 processing and RDKit for
the chemical-similarity fallback.

`Figure 4_P1.Rmd` can use the optional Bioconductor package `ChemmineR` for
the R-based chemical-similarity calculation. It is not available as a
compatible package in the configured Conda channels. Install it after
activating the environment if that branch is needed:

```bash
Rscript -e 'BiocManager::install("ChemmineR", ask=FALSE, update=FALSE)'
```

If `ChemmineR` is not installed, the workflow exports the required SMILES
table and can use the documented RDKit-based fallback instead.

## Reproduce the analyses

Run the commands from the repository root. The scripts resolve their input
and output locations from the repository layout, so the working directory is
not used as an implicit data location.

### Chapter 1

```bash
Rscript -e 'rmarkdown::render("scripts/Code_chapter01/Figure 1&S1.Rmd", output_dir="results/Code_chapter01")'
```

### Chapter 2

```bash
Rscript -e 'rmarkdown::render("scripts/Code_chapter02/Figure 2&S2.Rmd", output_dir="results/Code_chapter02")'
```

### Chapter 3

Run the Python scripts in order. P2 uses the UMAP and clustering outputs from
P1; P3 and P4 perform the two cell-line biology analyses; P5 compares their
enrichment results.

```bash
python "scripts/Code_chapter03/Figure 3_P1.py"
python "scripts/Code_chapter03/Figure 3_P2.py"
python "scripts/Code_chapter03/Figure 3_P3.py"
python "scripts/Code_chapter03/Figure 3_P4.py"
python "scripts/Code_chapter03/Figure 3_P5.py"
```

### Chapter 4

Run Chapter 3 P1 before `Figure 4D_P2.py`, because the latter uses the
Chapter 3 cluster and UMAP outputs. The supplementary R Markdown workflows
can be run after the data package is available.

```bash
Rscript -e 'rmarkdown::render("scripts/Code_chapter04/Figure 4_P1.Rmd", output_dir="results/Code_chapter04")'
python "scripts/Code_chapter04/Figure 4D_P2.py"
Rscript -e 'rmarkdown::render("scripts/Code_chapter04/Figure S5C-G_Jurkat_P3.Rmd", output_dir="results/Code_chapter04")'
Rscript -e 'rmarkdown::render("scripts/Code_chapter04/Figure S5D_L1000_P4.Rmd", output_dir="results/Code_chapter04")'
```

### Chapter 5

```bash
Rscript -e 'rmarkdown::render("scripts/Code_chapter05/Figure 5A.Rmd", output_dir="results/Code_chapter05")'
```

### Chapter 6

Run the projection and validation analyses before the downstream summary
analyses. `Figure S8G_P6.R` uses the concentration-model parameters produced
by Chapter 4 P1; either keep that result from the Zenodo package or run
Chapter 4 P1 first.

```bash
python "scripts/Code_chapter06/Figure S8A_P1.py"
python "scripts/Code_chapter06/Figure 6A_P2.py"
python "scripts/Code_chapter06/Figure S8B_P3.py"
python "scripts/Code_chapter06/Figure 6B&S8D_P4.py"
python "scripts/Code_chapter06/Figure 6C-E_P5.py"
Rscript "scripts/Code_chapter06/Figure S8G_P6.R"
```

Generated figures, tables, intermediate files, and reports are written below
`results/Code_chapterXX/`. Running a workflow may replace files with the same
name, so keep a backup of the original Zenodo results if you want to preserve
both the supplied and newly generated outputs.

## Check the installation

Before starting a full analysis, run the lightweight path and syntax check:

```bash
python scripts/validate_paths.py
```

This check confirms that the expected `data/`, `results/`, and `scripts/`
directories exist, parses the Python scripts, and checks that the workflows do
not contain machine-specific absolute paths. It does not replace a complete
analysis run; the L1000 GCTX file and the drug-profile directories are large
inputs and are still required.

## Notes for users

- Use the data and results from the same Zenodo release as the Git code. The
  directory layout and filenames are part of the workflow interface.
- Some analyses are computationally intensive and may require substantial
  memory and storage, particularly the L1000 and drug-profile workflows.
- Input datasets and third-party resources may have their own attribution and
  licensing requirements. Check the Zenodo metadata and the original data
  providers before redistributing them.
- The small README files in `data/` and `results/` describe the purpose of
  those packages when they are downloaded separately.

## License

The original analysis code in this repository is released under the MIT
License. See [LICENSE](LICENSE). The MIT License does not override the
license or usage terms of third-party input data distributed through Zenodo.
