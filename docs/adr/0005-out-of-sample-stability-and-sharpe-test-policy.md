# ADR-0005: Out-Of-Sample Stability And Sharpe Test Policy (Scope 7)

- Status: Accepted (current baseline)
- Date: 2026-03-01
- Scope: Scope 7 (`fin41360/workflows.py`, `fin41360/sharpe_tests.py`)

## Context

Scope 7 compares in-sample and out-of-sample portfolio performance. Team
implementations can diverge on test choice, assumptions, and interpretation.

## Decision (Implemented Now)

1. Portfolio comparison framing:
- Evaluate fixed IS-selected portfolios in OOS and compare against OOS-optimal references.

2. Sharpe ratio tests:
- Run **Jobson-Korkie** test (as implemented, Jorion reference form).
- Run **Ledoit-Wolf bootstrap** robust test (`n_boot=2000`) in parallel.

3. Frontier retention check:
- Use frontier replication metric (`frontier_replication_alpha`) to assess whether
  IS portfolios remain close to OOS efficient frontier combinations.

4. Workflow robustness:
- Scope 7 computes required intermediates internally to reduce notebook-order fragility.

## Rationale

- JK provides continuity with course references and classic literature.
- LW bootstrap relaxes strong distributional assumptions and improves robustness.
- Frontier replication adds geometric interpretation beyond p-values.

## Revisit Triggers And Tests

1. Trigger: JK and LW conclusions frequently diverge.
- Test: track sign/agreement rates across evaluated portfolios.
- Revisit threshold: persistent disagreement suggesting assumption mismatch.

2. Trigger: bootstrap uncertainty too high.
- Test: rerun LW with higher `n_boot` (e.g., 5000) and compare p-value stability.
- Revisit threshold: p-values highly unstable under reasonable bootstrap increases.

3. Trigger: weak OOS frontier retention despite IS strength.
- Test: monitor replication metrics and OOS tangency gaps.
- Revisit threshold: repeated low replication quality for selected IS portfolios.

## Consequences

- Scope 7 has a clear dual-test baseline (classic + robust).
- Statistical and geometric diagnostics are both used for interpretation.
- Method updates can be evaluated against an explicit baseline rather than ad-hoc notebook edits.
