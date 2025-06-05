library(quantmod)
library(xts)
library(dplyr)
library(lubridate)
library(copula)


symbols <- c("SPY", "EWJ", "EWG")  # SP500, Japan, Germany
names(symbols) <- c("SP500", "Nikkei", "DAX")

getSymbols(symbols, from = Sys.Date() - years(10), auto.assign = TRUE)

prices <- merge(
  Ad(SPY),
  Ad(EWJ),
  Ad(EWG)
)
colnames(prices) <- c("SP500", "Nikkei", "DAX")

returns <- na.omit(diff(log(prices)))
View(returns)

u_data <- pobs(as.matrix(returns))
head(u_data)

fit_t_full <- fitCopula(tCopula(dim = 3, dispstr = "un", df.fixed = FALSE), data = u_data, method = "ml")


summary(fit_t_full)

getSigma(fit_t_full@copula)

pairs(u_data, pch = 19, col = rgb(0,0,0,0.3), main = "Pseudo-Observations")

rho_12 <- getSigma(fit_t_full@copula)[1, 2]
df_t   <- coef(fit_t_full)["df"]

cop_12 <- ellipCopula(family = "t",
                      param  = rho_12,
                      dim    = 2,
                      df     = df_t)

contour(cop_12,
        FUN  = dCopula,
        main = "t-Copula Contour: SP500 vs Nikkei",
        xlab = expression(u[1]),
        ylab = expression(u[2]))

tail_dep <- tailIndex(fit_t_full@copula)
tail_dep

fit_gauss <- fitCopula(normalCopula(dim = 3, dispstr = "un"), data = u_data, method = "ml")
fit_clay  <- fitCopula(claytonCopula(dim = 3), data = u_data, method = "ml")
fit_gumb  <- fitCopula(gumbelCopula(dim = 3), data = u_data, method = "ml")

logLik(fit_t_full)
logLik(fit_gauss)
logLik(fit_clay)
logLik(fit_gumb)

# Summary of Copula-Based Dependence Analysis (SP500, Nikkei, DAX)
#
# Data: Daily log returns for SPY, EWJ, and EWG (3 years)
# Transformation: Empirical ranks converted to pseudo-observations

# Copula Fit:
# - Fitted multivariate t-copula with unstructured correlation matrix
# - Estimated df ≈ 7.7 indicating moderate tail thickness

# Dependence Structure:
# - Estimated correlations: ρ ≈ 0.68–0.73 across all pairs
# - Tail dependence coefficients (lower/upper): ~0.33–0.38
# - Indicates strong co-crash risk, symmetric across tails

# Model Comparison (Log-Likelihood):
# - t-Copula:       2241.05   ← best fit
# - Gaussian Copula: 2125.65   (no tail dependence)
# - Clayton:        1752.76   (lower-tail only, worse fit)
# - Gumbel:         1966.76   (upper-tail only, worse fit)

# Conclusion: t-copula best captures joint market risk behavior
# across SP500, Nikkei, and DAX. Gaussian and Archimedean copulas
# fail to capture observed symmetric tail dependence.


