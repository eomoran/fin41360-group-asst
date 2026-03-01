# ADR-0003: Bayes-Stein Mean Shrinkage And Covariance Shrinkage Policy

- Status: Accepted (current baseline), with explicit alternatives recorded
- Date: 2026-03-01
- Scope: Scope 2-3 (`fin41360/bayes_stein.py`, `fin41360/workflows.py`)

## Context

Different notebook variants used different Bayes-Stein targets and covariance
shrinkage methods, producing different shrinkage intensities and frontiers.
This ADR fixes the current baseline and records when to switch methods.

## Decision (Implemented Now)

1. Mean shrinkage target:
- Use Jorion-style Bayes-Stein shrinkage toward the **cross-sectional grand mean**.
- Implementation: `bayes_stein_means(mu, Sigma, T)` in `fin41360/bayes_stein.py`.

2. Covariance shrinkage:
- Use fixed convex shrinkage to scaled identity:
  `Sigma_bs = (1 - lambda) * Sigma + lambda * (trace(Sigma)/N) I`.
- Current default `lambda = 0.1` in Scope 2/3 workflows (`cov_shrink`).

3. Numerical safeguards:
- If `T <= N + 2`, skip mean shrinkage (`delta = 0`) to avoid unstable formula behavior.
- Use pseudo-inverse (`pinv`) for stability in frontier formulas.

## Rationale

- Cross-sectional target is transparent and consistent with current code and scope write-up.
- With high `T/N` in baseline industry sample, light covariance regularization is sufficient.
- Fixed `lambda` provides deterministic reproducibility across runs/team members.

## Alternatives Considered

1. Bayes-Stein target = GMV mean (`mu_gmv`):
- Valid and used in some team notebooks.
- Not current baseline; should be exposed as a parameter if we formalize dual-target runs.

2. Data-driven covariance shrinkage (e.g., Ledoit-Wolf estimator):
- Stronger statistical grounding for `lambda`.
- Not currently implemented for covariance estimation in Scope 2/3 baseline.

## Revisit Triggers And Tests

1. Trigger: shrinkage results too sensitive to fixed `lambda`.
- Test: lambda grid `{0.05, 0.10, 0.15}`.
- Revisit threshold: frontier ranking/tangency conclusions change qualitatively.

2. Trigger: unstable or extreme tangency weights.
- Test: report weight concentration and gross leverage `sum(abs(w))`.
- Revisit threshold: gross leverage > 5x or severe concentration in single names/factors.

3. Trigger: low effective sample regime.
- Test: monitor `T/(N+2)`.
- Revisit threshold: `T <= 3N` for scope universe (rule-of-thumb alert).

4. Trigger: OOS deterioration under current shrinkage.
- Test: compare IS/OOS Sharpe and frontier replication metrics in Scope 7 style checks.
- Revisit threshold: repeated large IS->OOS Sharpe collapse across estimators.

## Consequences

- Team has one documented baseline for Scope 2/3 comparisons.
- Differences vs teammate notebooks can now be attributed to explicit target/method choices.
- Transition to Ledoit-Wolf can be introduced as a controlled ADR update, not ad-hoc edits.
