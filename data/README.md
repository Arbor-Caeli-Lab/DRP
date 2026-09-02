# Data package

This directory contains the input data required by the analysis scripts. It is
intended to be distributed with the matching `results/` directory as a
separate Zenodo data package.

Keep the chapter-level directory names unchanged:

```text
data/
├── Code_chapter01/
├── Code_chapter02/
├── Code_chapter03/
├── Code_chapter04/
├── Code_chapter05/
└── Code_chapter06/
```

Chapter 2 uses `Data processing and Quality assessment.RData`, which is kept
inside `data/Code_chapter02/` because it is only used by that chapter.

The scripts locate this directory from their own file location, so the package
can be unpacked anywhere. The large L1000 GCTX file and the drug-profile
directories are intentionally kept here rather than in the Git code release.
