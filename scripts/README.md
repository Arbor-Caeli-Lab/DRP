# Reproduction scripts

Scripts are grouped by manuscript chapter under `scripts/Code_chapterXX/`.
Python scripts resolve `REPO_ROOT` from `__file__`; R/Rmd scripts use the
shared helper in `scripts/_common/repro_paths.R`. Consequently, paths do not
depend on the drive letter or on the caller's current working directory.
