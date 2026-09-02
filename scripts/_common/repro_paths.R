# Shared path helpers for the R/Rmd reproducibility scripts.
# The scripts live in scripts/Code_chapterXX, while data and results live in
# data/Code_chapterXX and results/Code_chapterXX respectively.

repro_script_dir <- function() {
  current_input <- tryCatch({
    if (requireNamespace("knitr", quietly = TRUE)) {
      knitr::current_input(dir = TRUE)
    } else {
      ""
    }
  }, error = function(e) "")

  if (length(current_input) == 1L && nzchar(current_input)) {
    return(normalizePath(current_input, winslash = "/", mustWork = FALSE))
  }

  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grepl("^--file=", args)]
  if (length(file_arg) > 0L) {
    script_file <- sub("^--file=", "", file_arg[[1]])
    return(normalizePath(dirname(script_file), winslash = "/", mustWork = FALSE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

repro_paths <- function(chapter) {
  if (!grepl("^Code_chapter[0-9]{2}$", chapter)) {
    stop("chapter must look like 'Code_chapter01'")
  }

  script_dir <- repro_script_dir()
  repo_root <- normalizePath(
    file.path(script_dir, "..", ".."),
    winslash = "/",
    mustWork = FALSE
  )

  results_dir <- file.path(repo_root, "results", chapter)
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  list(
    repo_root = repo_root,
    input_dir = file.path(repo_root, "data", chapter),
    results_dir = results_dir
  )
}
