# ADR-0004: Frontier Solver And Constraint Policy

- Status: Accepted (current baseline)
- Date: 2026-03-01
- Scope: Scope 2-6 (`fin41360/mv_frontier.py`, `fin41360/workflows.py`)

## Context

There are two implementation paths for MV portfolios:

- closed-form (unconstrained),
- numerical optimization (constrained).

Mixed use without policy leads to confusion about whether differences are
mathematical or solver-related.

## Decision (Implemented Now)

1. Default frontier/portfolios:
- Use **closed-form unconstrained** formulas for:
  - GMV weights,
  - tangency weights,
  - efficient frontier curve.

2. Constrained cases only:
- Use numerical optimization (`SLSQP`) only when explicit constraints are required.
- Current use: constrained tangency (`w_i >= w_min`) in Scope 6 FF3 handling.

3. Numerical fallback:
- Use pseudo-inverse (`pinv`) in unconstrained formulas for near-singular covariance matrices.

## Rationale

- Closed-form is exact for unconstrained mean-variance setup and easier to audit.
- Optimization introduces solver tolerance/dependence and should be used only when needed.
- This split keeps baseline deterministic and interpretable.

## Revisit Triggers And Tests

1. Trigger: unconstrained results become numerically unstable.
- Test: condition number and min eigenvalue diagnostics.
- Revisit threshold: repeated ill-conditioned covariance warnings or pathological weights.

2. Trigger: practical portfolio feasibility concerns dominate.
- Test: compare unconstrained vs constrained tangency outcomes.
- Revisit threshold: unconstrained portfolios systematically infeasible for intended interpretation.

3. Trigger: solver reliability concerns in constrained workflows.
- Test: monitor optimization success flags and KKT feasibility checks.
- Revisit threshold: non-trivial rate of optimization failures.

## Consequences

- Most scopes remain mathematically closed-form and reproducible.
- Constraint-driven scopes are explicit about optimization use and bounds.
- Team discussions can separate "model choice" from "solver choice" cleanly.
