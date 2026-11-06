suppressPackageStartupMessages({
  library(GSVA)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
expression_file <- if (length(args) >= 1) args[[1]] else "phase1/output/cell_expression.csv"
gmt_file <- if (length(args) >= 2) args[[2]] else "data/hallmark.gmt"
output_file <- if (length(args) >= 3) args[[3]] else "phase2/output/cell_gsva.csv"

expression_df <- read.csv(expression_file, check.names = FALSE, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
if (!identical(names(expression_df)[1:2], c("cell_id", "cancer_type"))) {
  stop("Expected cell_id and cancer_type as the first two columns")
}
cell_id <- expression_df$cell_id
cancer_type <- expression_df$cancer_type
expression_matrix <- t(as.matrix(expression_df[, -(1:2), drop = FALSE]))
storage.mode(expression_matrix) <- "double"
rownames(expression_matrix) <- names(expression_df)[-(1:2)]
colnames(expression_matrix) <- cell_id

gmt_lines <- readLines(gmt_file, warn = FALSE)
gene_sets <- lapply(strsplit(gmt_lines, "\t", fixed = FALSE), function(fields) unique(fields[-c(1, 2)]))
names(gene_sets) <- vapply(strsplit(gmt_lines, "\t", fixed = FALSE), `[[`, character(1), 1)

parameters <- gsvaParam(
  expression_matrix,
  gene_sets,
  minSize = 10,
  maxSize = Inf,
  kcdf = "Gaussian",
  tau = 1,
  maxDiff = TRUE,
  absRanking = FALSE,
  sparse = FALSE,
  filterRows = TRUE,
  checkNA = "yes",
  verbose = TRUE
)
scores <- gsva(parameters, BPPARAM = SerialParam(progressbar = FALSE), verbose = TRUE)

if (nrow(scores) != length(gene_sets) || ncol(scores) != length(cell_id)) {
  stop(sprintf("Unexpected GSVA result dimensions: %d x %d", nrow(scores), ncol(scores)))
}

output_df <- data.frame(
  cell_id = cell_id,
  cancer_type = cancer_type,
  t(scores),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(output_df, output_file, row.names = FALSE, quote = TRUE)
cat(sprintf("GSVA_OK pathways=%d cells=%d\n", nrow(scores), ncol(scores)))
