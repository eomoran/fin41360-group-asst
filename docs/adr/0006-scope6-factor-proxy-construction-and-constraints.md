# ADR-0006: Scope 6 Factor-Proxy Construction And Constraint Policy

- Status: Accepted (current baseline)
- Date: 2026-03-01
- Scope: Scope 6 (`fin41360/workflows.py`, `fin41360/mv_frontier.py`)

## Context

Scope 6 compares FF factor frontiers to implementable proxy frontiers. Results
can diverge materially depending on proxy construction and tangency constraints.

## Decision (Implemented Now)

1. Return space:
- Run Scope 6 in **excess-return space** (`rf=0` in optimization formulas).
- Proxy returns are converted to excess by subtracting aligned RF.

2. Proxy model sets:
- Proxy-3 uses `["Mkt", "SMB", "HML"]`.
- Proxy-5 uses `["Mkt", "SMB", "HML", "RMW", "CMA"]`.

3. Constraint policy:
- FF3 tangency is computed with explicit lower bound `w_i >= -1.0` (SLSQP),
  to avoid extreme unconstrained leverage.
- Other unconstrained frontiers/GMV points remain closed-form baseline unless
  explicit constraints are requested.

4. Alignment policy:
- Use intersection of dates across FF3, FF5, proxies, and RF.

## Rationale

- Excess-return space ensures comparability across FF and proxy sets.
- Constrained FF3 tangency improves interpretability of charts and summaries.
- Date intersection avoids accidental look-ahead/misaligned panel comparisons.

## Revisit Triggers And Tests

1. Trigger: constrained FF3 tangency materially changes conclusions.
- Test: compare constrained vs unconstrained FF3 tangency Sharpe and leverage.
- Revisit threshold: qualitative ranking changes solely from constraint choice.

2. Trigger: proxy mapping instability.
- Test: perturb proxy universe/mapping and re-run Scope 6 comparison.
- Revisit threshold: headline conclusions highly sensitive to small mapping edits.

3. Trigger: overly short overlap period.
- Test: report `n_obs` and stress-test over alternative windows when available.
- Revisit threshold: overlap too short for stable covariance estimates.

## Consequences

- Scope 6 is explicit about when optimization constraints are applied.
- Proxy comparison is reproducible and auditable by teammates.
- Method drift from chatbot-generated variants is reduced.
