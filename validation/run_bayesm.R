# Fit bayesm::rhierMnlRwMixture to the camera data and write summaries as CSV.
# Called by compare_camera.py in a separate process; bayesm (GPL) is never imported
# into partworth.
#
# Usage: Rscript run_bayesm.R <out_dir> <n_estimation_tasks> <R> <keep> <seed>
args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
n_est <- as.integer(args[2])
R <- as.integer(args[3])
keep <- as.integer(args[4])
set.seed(as.integer(args[5]))

suppressPackageStartupMessages(library(bayesm))
data(camera)
p <- 5  # four cameras plus the outside option
lgt <- lapply(camera, function(r) {
  rows <- seq_len(n_est * p)
  list(y = r$y[seq_len(n_est)], X = r$X[rows, , drop = FALSE])
})
fit <- rhierMnlRwMixture(
  Data = list(lgtdata = lgt, p = p),
  Prior = list(ncomp = 1),
  Mcmc = list(R = R, keep = keep, nprint = 0)
)
burn <- floor(dim(fit$betadraw)[3] / 2)
draws <- fit$betadraw[, , (burn + 1):dim(fit$betadraw)[3], drop = FALSE]
cols <- colnames(camera[[1]]$X)
beta_mean <- apply(draws, c(1, 2), mean)
colnames(beta_mean) <- cols
write.csv(beta_mean, file.path(out_dir, "bayesm_beta_mean.csv"), row.names = FALSE)

mu <- t(sapply((burn + 1):length(fit$nmix$compdraw), function(i) fit$nmix$compdraw[[i]][[1]]$mu))
colnames(mu) <- cols
write.csv(mu, file.path(out_dir, "bayesm_mu_draws.csv"), row.names = FALSE)
cat("bayesm", as.character(packageVersion("bayesm")), "\n",
    file = file.path(out_dir, "bayesm_version.txt"))
