# FIN41360 Assignment 1: Report Format and Brief Notes

## Non-negotiable submission requirements
- Deadline: Thursday, March 5, 2026 at 5:00pm (soft copy on Brightspace + hard copy in Smurfit Foyer mailbox).
- Max length: 3000 words, including tables/figures/everything in report body.
- Bibliography is excluded from the word count.
- Submission form must be attached and fully completed (assignment number, team number, names/student numbers, individual contributions).
- Workings/data/code must also be emailed to `emmanuel.eyiah-donkor@ucd.ie` with subject:
  - `FIN41360: Assignment 1 (Group X)`

## Required analysis structure from the brief
- Task 1: Review Ken French data library metadata/legends.
- Task 2: 30 industries (1980-01 to 2025-12), sample frontier vs Bayes-Stein variants, GMV/tangency metrics.
- Task 3: One stock per industry, matched sample window vs industry frontier comparison.
- Task 4: Add risk-free asset to 30-industry opportunity set and compare.
- Task 5: Replace industries with FF3 then FF5 factors; compare in excess-return space.
- Task 6: Replace factors with investable practical proxies; compare same-window frontiers.
- Task 7: Two contiguous subperiods ending Dec-2002 and Dec-2025; IS vs OOS Sharpe testing using Jobson-Korkie and Ledoit-Wolf.
- Task 8: Extension under additional method/constraint (e.g., resampling, GMV focus, diversification change).
- Task 9: Evaluate 3 mutual funds vs market/proxy/tangency benchmarks.

## Report-writing guidance explicitly stated in the brief
- Prioritize clarity over coverage; unclear discussion is penalized heavily.
- If short on time: do earlier tasks well rather than all tasks poorly.
- Motivate any modeling/implementation choices using module theory and references.
- Include literature discussion where relevant, succinct but systematic.
- Use an academic-paper style layout (or equally rigorous industry-report style).
- Design tables/figures to maximize immediate readability and interpretation.
- Condense findings into as few high-quality exhibits as possible.
- Appoint an editor to enforce coherence and flow across sections.

## Suggested structure in this repo
- Main draft template: `report/main.tex`
- References file: `report/references.bib`
- Supporting policy/decision notes: `docs/adr/*.md`
- Scope-specific mapping notes: `docs/scope3_ff30_mapping_method.md`

## Immediate checklist for draft completion
- Lock final sample windows and ensure apples-to-apples comparisons by task.
- Generate one primary figure/table per task before writing long prose.
- Keep each task write-up to: setup, core findings, economic interpretation.
- Track live word count from first full draft to stay below 3000.
- Add contribution statement consistent with submission form.
