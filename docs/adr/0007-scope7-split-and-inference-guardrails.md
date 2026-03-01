# ADR-0007: Scope 7 Split Design And Inference Guardrails

- Status: Accepted (current baseline; intentionally low-parameter)
- Date: 2026-03-01
- Scope: Scope 7 (`fin41360/workflows.py`, `fin41360/sharpe_tests.py`)

## Context

Scope 7 is primarily an inference/stability exercise (IS vs OOS), not a tuning
exercise. Over-parameterization here can invalidate interpretation.

## Decision (Implemented Now)

1. Keep Scope 7 low-parameter:
- Use fixed contiguous IS/OOS split design as implemented in workflow.
- Avoid additional estimator-selection knobs inside Scope 7 baseline runs.

2. Inference stack:
- Report both JK and Ledoit-Wolf Sharpe-equality tests.
- Include frontier-replication diagnostics for geometric interpretation.

3. Reproducibility:
- Scope 7 computes dependent quantities internally to avoid notebook-order drift.

## Rationale

- Scope 7 should test robustness of selected portfolios, not optimize extra knobs.
- Dual-test reporting (classic + robust) balances interpretability and robustness.
- Guarding against hidden notebook state is critical for team reproducibility.

## Revisit Triggers And Tests

1. Trigger: split dependence concern.
- Test: run at least one alternate contiguous split as robustness appendix.
- Revisit threshold: conclusions reverse under nearby plausible split.

2. Trigger: test disagreement concern.
- Test: compare JK vs LW agreement and bootstrap stability (`n_boot` sensitivity).
- Revisit threshold: persistent disagreement with unstable bootstrap inference.

3. Trigger: OOS frontier retention concerns.
- Test: monitor replication and OOS tangency gap for fixed IS portfolios.
- Revisit threshold: repeated large retention failures despite strong IS fit.

## Consequences

- Scope 7 remains defensible as an evaluation scope rather than a tuning scope.
- Teammates can explain why fewer “method knobs” are exposed here.
