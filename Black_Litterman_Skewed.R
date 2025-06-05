library(quantmod)
library(dplyr)
library(xts)
library(moments)
library(lubridate)
library(sn)
library(MASS)

tickers <- c("SPY", "TLT", "QQQ", "GLD", "EEM")
getSymbols(tickers, from = Sys.Date() - years(5), periodicity = "monthly", auto.assign = TRUE)

prices <- do.call(merge, lapply(tickers, function(sym) Ad(get(sym))))
colnames(prices) <- tickers

returns <- na.omit(diff(log(prices)))
Sigma   <- cov(returns)

w_mkt <- c(0.35, 0.20, 0.20, 0.15, 0.10)
w_mkt <- w_mkt / sum(w_mkt)
lambda     <- 2.5
mu_market  <- lambda * Sigma %*% w_mkt   # 5×1

skew_vec <- apply(returns, 2, skewness)  # length 5
alpha_0  <- 5 * skew_vec                 # length 5

P <- matrix(c(
  0, -1,  1,  0,  0,   # “QQQ – TLT = 3%”
  0,  0,  0, -1,  1,   # “EEM – GLD = 2%”
  1,  0,  0,  0,  0    # “SPY = 7%”
), nrow = 3, byrow = TRUE)

q       <- c(0.03, 0.02, 0.07)  # view means (3×1)
alpha_1 <- c(5, 4, 6)           # skew for each view (length 3)

tau <- 0.025
Omega <- diag(diag(P %*% Sigma %*% t(P))) * tau  # 3×3 diagonal

log_posterior <- function(mu) {
  # Prior: mu ~ SN( mu_market, tau*Sigma, alpha_0 )
  prior_log <- dmsn(
    x     = mu,
    xi    = as.numeric(mu_market),
    Omega = tau * Sigma,
    alpha = alpha_0,
    log   = TRUE
  )

  # View: P %*% mu ~ SN( q, Omega, alpha_1 )
  view_error <- as.numeric(P %*% mu)  # length = 3
  view_log   <- dmsn(
    x     = view_error,
    xi    = q,
    Omega = Omega,
    alpha = alpha_1,
    log   = TRUE
  )

  return(prior_log + view_log)
}

metropolis_sampler <- function(log_post, init, n_iter = 20000, prop_cov, burn = 5000) {
  mu_dim         <- length(init)
  samples        <- matrix(NA, nrow = n_iter, ncol = mu_dim)
  current        <- init
  current_lpost  <- log_post(current)
  accept_count   <- 0

  for (i in 1:n_iter) {
    proposal       <- MASS::mvrnorm(1, mu = current, Sigma = prop_cov)
    proposal_lpost <- log_post(proposal)
    log_accept     <- proposal_lpost - current_lpost
    if (log(runif(1)) < log_accept) {
      current       <- proposal
      current_lpost <- proposal_lpost
      accept_count  <- accept_count + 1
    }
    samples[i, ] <- current
  }
  cat("Acceptance Rate:", accept_count / n_iter, "\n")
  return(samples[(burn + 1):n_iter, ])  # drop burn‐in
}

set.seed(42)
prop_cov_tuned <- (0.05)^2 * diag(diag(Sigma))    # proposal on mu (5×5 diagonal)
init     <- as.numeric(mu_market)      # start at implied means

posterior_samples_tuned <- metropolis_sampler(
  log_post = log_posterior,
  init     = init,
  n_iter   = 20000,
  prop_cov = prop_cov_tuned,
  burn     = 5000
)

mu_summary <- data.frame(
  Asset = colnames(prices),
  Mean  = colMeans(posterior_samples_tuned),
  Lower = apply(posterior_samples_tuned, 2, quantile, probs = 0.025),
  Upper = apply(posterior_samples_tuned, 2, quantile, probs = 0.975)
)
print(mu_summary)

inv_Sigma <- solve(Sigma)

get_weights <- function(mu) {
  raw <- inv_Sigma %*% mu
  as.numeric(raw / sum(raw))  # normalize so weights sum to 1
}


weights_matrix <- t(apply(posterior_samples_tuned, 1, get_weights))


weight_summary <- data.frame(
  Asset = colnames(prices),
  Mean  = colMeans(weights_matrix),
  Lower = apply(weights_matrix, 2, quantile, probs = 0.025),
  Upper = apply(weights_matrix, 2, quantile, probs = 0.975)
)

print(weight_summary)


# Portfolio Weights Interpretation (Bayesian Black-Litterman)
#
# - Weights were computed using the unconstrained mean-variance formula:
#     w* = (Sigma^{-1} * mu) / (1ᵗ * Sigma^{-1} * mu)
#
# - This allows for negative weights → i.e., short positions
# - Interpretation:
#     - Negative weight: short the asset (bet it will underperform)
#     - Positive weight: long the asset (standard exposure)
#
# - In practice, shorting may be disallowed (e.g., retirement funds, ETFs)
# - To enforce no shorting, use constrained optimization:
#     - Maximize expected return – variance
#     - Subject to: weights ≥ 0 and sum to 1
#     - Use solve.QP() from quadprog for this
#
# - Current setup reflects theoretical optimal portfolio under full flexibility.


