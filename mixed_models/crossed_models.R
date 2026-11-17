# Supplementary Table 3: the main tests with participants, problems and questions as random effects,
# so that the uncertainty also reflects the sampling of questions and problems.
# Run from the repository root after export_long_data.py:  Rscript mixed_models/crossed_models.R
# Requires the R packages lme4 and lmerTest.
suppressMessages({library(lme4); library(lmerTest)})
load_test <- function(file, test) {
  d <- read.csv(file.path("mixed_models/data", file))
  data.frame(test = test, participant = paste0(test, ":", d$participant), question = paste0(test, ":", d$question),
             problem = as.character(d$problem),          # FOLIO problems shared by the test and its control keep one identifier
             r1 = as.integer(grepl("DeepSeek", d$model)), correct = d$correct, rt = d$rt)
}
gsm <- load_test("gsm8k_long.csv", "GSM8K"); fol <- load_test("folio_long.csv", "FOLIO")
ctl <- load_test("folio_control_long.csv", "FOLIO control")
RE <- "(1 | participant) + (1 | problem) + (1 | question)"
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
out <- list()
add <- function(...) out[[length(out) + 1]] <<- data.frame(...)

# Mean accuracy vs chance: linear mixed model of correctness (in percent), Satterthwaite df, H0: mean = 25.
for (d in list(gsm, fol)) {
  m <- lmer(as.formula(paste("I(100 * correct) ~ 1 +", RE)), d)
  co <- summary(m)$coefficients; est <- co[1, 1]; se <- co[1, 2]; df <- co[1, 3]; t <- (est - 25) / se
  add(test = paste(d$test[1], "accuracy vs 25% chance"), estimate = est, ci_low = est - qt(.975, df) * se,
      ci_high = est + qt(.975, df) * se, p = 2 * pt(-abs(t), df), measure = "mean accuracy (%)")
}
# Comparisons and response time: logistic mixed models, odds ratios with Wald 95% CIs.
glmm <- function(d, x, label, measure) {
  m <- glmer(as.formula(paste("correct ~", x, "+", RE)), d, binomial, control = ctrl)
  co <- summary(m)$coefficients[x, ]
  add(test = label, estimate = exp(co[1]), ci_low = exp(co[1] - qnorm(.975) * co[2]),
      ci_high = exp(co[1] + qnorm(.975) * co[2]), p = co[4], measure = measure)
}
glmm(gsm, "r1", "GSM8K: DeepSeek-R1 vs Qwen3-32B items", "odds ratio")
glmm(fol, "r1", "FOLIO: DeepSeek-R1 vs Qwen3-32B items", "odds ratio")
fc <- rbind(fol, ctl); fc$control <- as.integer(fc$test == "FOLIO control")
glmm(fc, "control", "FOLIO control vs FOLIO test", "odds ratio")
# Response time: the filter of the main analysis (under 5 s or above Q3 + 3 x IQR set aside), standardized response time.
for (d in list(gsm, fol, ctl)) {
  v <- d[!is.na(d$rt), ]; q <- quantile(v$rt, c(.25, .75)); k <- v[v$rt >= 5 & v$rt <= q[2] + 3 * (q[2] - q[1]), ]
  k$rt_sd <- (k$rt - mean(k$rt)) / sd(k$rt)
  glmm(k, "rt_sd", paste(d$test[1], "response time"), sprintf("odds ratio per s.d. (%d responses)", nrow(k)))
}
res <- do.call(rbind, out)
res[, c("estimate", "ci_low", "ci_high")] <- round(res[, c("estimate", "ci_low", "ci_high")], 2)
res$p <- signif(res$p, 2)
print(res, row.names = FALSE)
# Sensitivity check reported in the legend: a by-participant random slope for model.
for (d in list(gsm, fol)) {
  m <- glmer(correct ~ r1 + (1 + r1 | participant) + (1 | problem) + (1 | question), d, binomial, control = ctrl)
  co <- summary(m)$coefficients["r1", ]
  cat(sprintf("%s, with a by-participant random slope for model: odds ratio %.2f, p = %.2g (singular fit: %s)\n",
              d$test[1], exp(co[1]), co[4], isSingular(m)))
}
