# Expert study v6.2 (prereg v3, osf.io/fspuc): the confirmatory mixed models.
#   Rscript models.R <dir with pairs.csv, anchors.csv> <out.json> <config.json>
# R 4.4.1, lme4 1.1-35.5, lmerTest 3.1-3 (docker image hots-r-lme4:4.4.1).
# glmer: Laplace approximation (nAGQ = 1), default optimizer; Wald CIs/p.
# Convergence fallback chains exactly as in prereg §§4-6. A fit counts as
# failed if it errors or lme4 reports a convergence message
# (optinfo$conv$lme4$messages); a singular fit is not a convergence failure
# (it is recorded).
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(jsonlite); library(sandwich); library(lmtest)
})
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; out <- args[2]
cfg <- fromJSON(args[3])
set.seed(cfg$seed)
P <- read.csv(file.path(dir, "pairs.csv"), stringsAsFactors = FALSE)
A <- read.csv(file.path(dir, "anchors.csv"), stringsAsFactors = FALSE)
P$rater <- factor(P$rater); P$item <- factor(P$item)
A$rater <- factor(A$rater); A$item <- factor(A$item)

conv_msgs <- function(fit) {
  # lme4 lists a singular (boundary) fit among its convergence messages; a
  # singular fit is a valid estimate, so it is recorded but is not a failure.
  m <- fit@optinfo$conv$lme4$messages
  if (is.null(m)) character(0) else m[!grepl("singular", m)]
}
try_fit <- function(expr) {
  w <- character(0)
  fit <- tryCatch(withCallingHandlers(expr, warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") }),
                  error = function(e) e)
  if (inherits(fit, "error")) return(list(ok = FALSE, error = conditionMessage(fit), warnings = w))
  msgs <- conv_msgs(fit)
  list(ok = length(msgs) == 0, fit = fit, conv = msgs, warnings = w, singular = isSingular(fit))
}
fixef_row <- function(fit, term) {
  s <- summary(fit)$coefficients
  est <- s[term, "Estimate"]; se <- s[term, "Std. Error"]
  z <- est / se
  list(term = term, est = est, se = se, z = z, p_two_sided = 2 * pnorm(-abs(z)),
       ci_lo = est - qnorm(0.975) * se, ci_hi = est + qnorm(0.975) * se)
}
plogis_list <- function(r) list(est = plogis(r$est), ci_lo = plogis(r$ci_lo), ci_hi = plogis(r$ci_hi))
wilson <- function(k, n) {
  if (n == 0) return(list(k = k, n = n))
  b <- binom.test(k, n)  # exact CI kept too
  z <- qnorm(0.975); ph <- k / n
  c0 <- (ph + z^2 / (2 * n)) / (1 + z^2 / n); h <- z * sqrt(ph * (1 - ph) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  list(k = k, n = n, rate = ph, wilson_lo = c0 - h, wilson_hi = c0 + h)
}

# ── H1-type agreement test (also S5 and the sensitivities) ─────────────
agree_test <- function(d, ycol) {
  d$y <- d[[ycol]]
  res <- list(n_judgments = nrow(d), n_items = length(unique(d$item)), n_raters = length(unique(d$rater)),
              pooled = wilson(sum(d$y), nrow(d)))
  f1 <- try_fit(glmer(y ~ 1 + (1 | rater) + (1 | item), data = d, family = binomial))
  if (f1$ok) {
    r <- fixef_row(f1$fit, "(Intercept)")
    res$model <- "glmer y ~ 1 + (1|rater) + (1|item)"; res$singular <- f1$singular
    res$intercept <- r; res$prob_scale <- plogis_list(r)
    res$success <- r$ci_lo > 0
    return(res)
  }
  res$fallback_reason_1 <- c(f1$conv, f1$error)
  f2 <- try_fit(glmer(y ~ 1 + (1 | item), data = d, family = binomial))
  if (f2$ok) {
    r <- fixef_row(f2$fit, "(Intercept)")
    res$model <- "fallback (i): glmer y ~ 1 + (1|item)"; res$singular <- f2$singular
    res$intercept <- r; res$prob_scale <- plogis_list(r); res$success <- r$ci_lo > 0
    return(res)
  }
  res$fallback_reason_2 <- c(f2$conv, f2$error)
  mv <- aggregate(y ~ item, d, function(v) mean(v))
  mv <- mv[mv$y != 0.5, ]   # items whose ratings split evenly carry no majority
  k <- sum(mv$y > 0.5); n <- nrow(mv)
  bt <- binom.test(k, n, 0.5)
  res$model <- "fallback (ii): exact binomial on per-item majority votes (even splits dropped)"
  res$majority <- list(k = k, n = n, p_two_sided = bt$p.value, prop = k / n)
  res$success <- bt$p.value < 0.05 && k / n > 0.5
  res
}

# ── H2 moderation ───────────────────────────────────────────────────────
classify_h2 <- function(lo, hi, margin) {
  if (hi < 0) return(if (lo >= margin) "OOD-degradation (bounded)" else "OOD-degradation")
  if (lo >= margin) return("Robust (non-inferiority met)")
  "Inconclusive"
}
h2_test <- function(d, covar) {
  it <- unique(d[, c("item", covar)])
  mu <- mean(it[[covar]]); sdv <- sd(it[[covar]])
  d$ood <- (d[[covar]] - mu) / sdv
  d$y <- d$agree_cons
  res <- list(covariate = covar, n_judgments = nrow(d), n_items = nrow(it), z_mean = mu, z_sd = sdv)
  f1 <- try_fit(glmer(y ~ ood + (1 | rater) + (1 | item), data = d, family = binomial))
  f <- if (f1$ok) f1 else NULL
  if (is.null(f)) {
    res$fallback_reason_1 <- c(f1$conv, f1$error)
    f2 <- try_fit(glmer(y ~ ood + (1 | item), data = d, family = binomial))
    if (f2$ok) { f <- f2; res$model <- "fallback (i): glmer y ~ ood + (1|item)" } else res$fallback_reason_2 <- c(f2$conv, f2$error)
  } else res$model <- "glmer y ~ ood + (1|rater) + (1|item)"
  if (!is.null(f)) {
    r <- fixef_row(f$fit, "ood"); res$singular <- f$singular; res$slope <- r
    res$class_confirmatory <- classify_h2(r$ci_lo, r$ci_hi, cfg$h2_margin)
    res$class_margin_020 <- classify_h2(r$ci_lo, r$ci_hi, cfg$h2_margin_sensitivity)
    return(res)
  }
  im <- aggregate(cbind(y, ood) ~ item, d, mean)
  rho <- cor(im$y, im$ood, method = "spearman")
  perm <- replicate(100000, cor(sample(im$y), im$ood, method = "spearman"))
  p <- (sum(abs(perm) >= abs(rho)) + 1) / (length(perm) + 1)
  res$model <- "fallback (ii): Spearman item mean agreement vs ood, 100,000 permutations"
  res$spearman <- list(rho = rho, p_two_sided = p)
  res$class_confirmatory <- if (p < 0.05 && rho < 0) "OOD-degradation (fallback)" else "No classification (fallback: robustness not claimable)"
  res
}

near <- function(d, col, thr) d[abs(d[[col]] - 0.5) > thr, ]
R <- list(meta = list(R = R.version.string, lme4 = as.character(packageVersion("lme4")),
                      lmerTest = as.character(packageVersion("lmerTest")),
                      n_pair_judgments = nrow(P), n_anchor_judgments = nrow(A),
                      raters_pairs = sort(unique(as.character(P$rater)))))

# H1 primary and sensitivities
R$H1 <- agree_test(near(P, "p_cons", 0.02), "agree_cons")
R$H1_sens_threshold <- lapply(c(t001 = 0.01, t002 = 0.02, t005 = 0.05), function(t) agree_test(near(P, "p_cons", t), "agree_cons"))
R$H1_sens_evaluator <- lapply(c(naive = "naive", herostrength = "herostrength", enriched = "enriched"),
                              function(e) agree_test(near(P, paste0("p_", e), 0.02), paste0("agree_", e)))
R$H1_uncorrected_descriptive <- agree_test(near(P, "p_cons_unc", 0.02), "agree_cons_unc")
H1set <- near(P, "p_cons", 0.02)
R$H1_patch257 <- list(a = agree_test(H1set[H1set$p257a == 0, ], "agree_cons"),
                      b = agree_test(H1set[H1set$p257b == 0, ], "agree_cons"))

# H2
R$H2 <- h2_test(H1set, "ood_var_max")
R$H2_sens_covariate <- list(ood_var_mean = h2_test(H1set, "ood_var_mean"), ood_var_matchup = h2_test(H1set, "ood_var_matchup"))
R$H2_patch257 <- list(a = h2_test(H1set[H1set$p257a == 0, ], "ood_var_max"),
                      b = tryCatch(h2_test(H1set[H1set$p257b == 0, ], "ood_var_max"), error = function(e) list(error = conditionMessage(e))))

# S5
S5set <- H1set[H1set$stratum == "vs_anchored", ]
R$S5 <- agree_test(S5set, "agree_cons")
R$S5_patch257 <- list(a = agree_test(S5set[S5set$p257a == 0, ], "agree_cons"),
                      b = tryCatch(agree_test(S5set[S5set$p257b == 0, ], "agree_cons"), error = function(e) list(error = conditionMessage(e))))

# S1 per tier, Holm family of 3
s1_tier <- function(d) {
  d$y <- d$correct
  res <- list(n_judgments = nrow(d), n_items = length(unique(d$item)), pooled = wilson(sum(d$y), nrow(d)))
  f1 <- try_fit(glmer(y ~ 1 + (1 | rater) + (1 | item), data = d, family = binomial))
  f <- if (f1$ok) f1 else NULL
  if (is.null(f)) {
    res$fallback_reason <- c(f1$conv, f1$error)
    f2 <- try_fit(glmer(y ~ 1 + (1 | item), data = d, family = binomial))
    if (f2$ok) { f <- f2; res$model <- "fallback: glmer y ~ 1 + (1|item) (the H1 chain; S1 has no pre-specified fallback)" }
  } else res$model <- "glmer y ~ 1 + (1|rater) + (1|item)"
  if (!is.null(f)) { r <- fixef_row(f$fit, "(Intercept)"); res$intercept <- r; res$prob_scale <- plogis_list(r); res$singular <- f$singular }
  res
}
s1_family <- function(d) {
  tiers <- c("low", "mid", "high")
  out <- lapply(setNames(tiers, tiers), function(t) s1_tier(d[d$tier == t, ]))
  ps <- sapply(tiers, function(t) if (!is.null(out[[t]]$intercept)) out[[t]]$intercept$p_two_sided else NA)
  adj <- p.adjust(ps, method = "holm")
  for (t in tiers) {
    out[[t]]$p_holm <- unname(adj[t])
    out[[t]]$above_chance <- !is.na(adj[t]) && adj[t] < 0.05 && out[[t]]$intercept$est > 0
  }
  out
}
R$S1 <- s1_family(A)
R$S1_patch257 <- list(a = s1_family(A[A$p257a == 0, ]), b = tryCatch(s1_family(A[A$p257b == 0, ]), error = function(e) list(error = conditionMessage(e))))

# S2 slider calibration on all rated machine pairs
s2 <- list(n_judgments = nrow(P))
f1 <- try_fit(lmer(slider0 ~ p_cons + (1 + p_cons | rater), data = P))
if (f1$ok) {
  co <- summary(f1$fit)$coefficients["p_cons", ]
  ci <- co["Estimate"] + c(-1, 1) * qt(0.975, co["df"]) * co["Std. Error"]
  s2 <- c(s2, list(model = "lmer slider0 ~ p_cons + (1 + p_cons | rater), Satterthwaite", singular = f1$singular,
                   slope = unname(co["Estimate"]), se = unname(co["Std. Error"]), df = unname(co["df"]),
                   p_two_sided = unname(co["Pr(>|t|)"]), ci_lo = ci[1], ci_hi = ci[2]))
} else {
  s2$fallback_reason_1 <- c(f1$conv, f1$error)
  f2 <- try_fit(lmer(slider0 ~ p_cons + (1 | rater), data = P))
  if (f2$ok) {
    co <- summary(f2$fit)$coefficients["p_cons", ]
    ci <- co["Estimate"] + c(-1, 1) * qt(0.975, co["df"]) * co["Std. Error"]
    s2 <- c(s2, list(model = "fallback (i): lmer slider0 ~ p_cons + (1 | rater), Satterthwaite", singular = f2$singular,
                     slope = unname(co["Estimate"]), se = unname(co["Std. Error"]), df = unname(co["df"]),
                     p_two_sided = unname(co["Pr(>|t|)"]), ci_lo = ci[1], ci_hi = ci[2]))
  } else {
    s2$fallback_reason_2 <- c(f2$conv, f2$error)
    m <- lm(slider0 ~ p_cons + factor(rater), data = P)
    ct <- coeftest(m, vcov = vcovCL(m, cluster = ~rater))
    s2 <- c(s2, list(model = "fallback (ii): rater fixed effects, cluster-robust (rater) SE",
                     slope = ct["p_cons", 1], se = ct["p_cons", 2], p_two_sided = ct["p_cons", 4],
                     ci_lo = ct["p_cons", 1] - 1.96 * ct["p_cons", 2], ci_hi = ct["p_cons", 1] + 1.96 * ct["p_cons", 2]))
  }
}
s2$positive_calibration <- !is.null(s2$p_two_sided) && s2$slope > 0 && s2$p_two_sided < 0.05
R$S2 <- s2

# S3 low-tier anchors, group A vs B, one-sided
L <- A[A$tier == "low", ]
L$groupA <- as.integer(L$group == "A")
s3 <- list(n_judgments = nrow(L), raters_A = sort(unique(as.character(L$rater[L$groupA == 1]))),
           raters_B = sort(unique(as.character(L$rater[L$groupA == 0]))),
           pooled_A = wilson(sum(L$correct[L$groupA == 1]), sum(L$groupA == 1)),
           pooled_B = wilson(sum(L$correct[L$groupA == 0]), sum(L$groupA == 0)))
f1 <- try_fit(glmer(correct ~ groupA + (1 | rater) + (1 | item), data = L, family = binomial))
f <- if (f1$ok) { s3$model <- "glmer correct ~ group + (1|rater) + (1|item)"; f1 } else NULL
if (is.null(f)) {
  s3$fallback_reason_1 <- c(f1$conv, f1$error)
  f2 <- try_fit(glmer(correct ~ groupA + (1 | item), data = L, family = binomial))
  if (f2$ok) { f <- f2; s3$model <- "fallback (i): glmer correct ~ group + (1|item)" } else s3$fallback_reason_2 <- c(f2$conv, f2$error)
}
if (!is.null(f)) {
  r <- fixef_row(f$fit, "groupA"); s3$effect <- r; s3$singular <- f$singular
  s3$p_one_sided <- pnorm(-r$z)
} else {
  ka <- sum(L$correct[L$groupA == 1]); na <- sum(L$groupA == 1); kb <- sum(L$correct[L$groupA == 0]); nb <- sum(L$groupA == 0)
  pp <- (ka + kb) / (na + nb); z <- (ka / na - kb / nb) / sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
  s3$model <- "fallback (ii): two-proportion z, one-sided (A > B); a deviation"
  s3$z <- z; s3$p_one_sided <- pnorm(-z)
}
s3$significant_one_sided <- s3$p_one_sided < 0.05
R$S3 <- s3

write(toJSON(R, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null"), out)
cat("models done ->", out, "\n")
