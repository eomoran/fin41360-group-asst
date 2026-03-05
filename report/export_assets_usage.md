# Exporting Notebook Assets for the LaTeX Report

Use these helpers in `fin41360/report_assets.py` to save plots/tables into:
- `report/figures/`
- `report/tables/`

By default files are **not overwritten**. If a name already exists, a version suffix is used (for example `_v02`).

## Notebook snippet

```python
from fin41360.report_assets import save_figures, save_scope_summary_tables, save_table

# --- figures ---
saved_figs = save_figures(
    {
        "task2_industries_frontiers": fig_scope2,
        "task3_with_coal_overlay": fig_scope3_with_coal,
        "task3_drop_coal_overlay": fig_scope3_drop_coal,
        "task4_industries_with_rf": fig_scope4,
        "task5_ind_vs_ff3_ff5": fig_scope5,
        "task6_ff_vs_proxy_overlay": fig_scope6_overlay,
        "task6_ff_vs_proxy_panels": fig_scope6_panels,
    },
    overwrite=False,   # keep prior exports by versioning
    fmt="png",
    dpi=220,
)
saved_figs
```

```python
# --- workflow summary tables ---
scope2_paths = save_scope_summary_tables(scope2, "scope2", overwrite=False)
scope3_with_paths = save_scope_summary_tables(scope3_with_coal, "scope3_with_coal", overwrite=False)
scope3_drop_paths = save_scope_summary_tables(scope3_drop_coal, "scope3_drop_coal", overwrite=False)
scope4_paths = save_scope_summary_tables(scope4, "scope4", overwrite=False)
scope5_paths = save_scope_summary_tables(scope5, "scope5", overwrite=False)
scope6_paths = save_scope_summary_tables(scope6, "scope6", overwrite=False)
scope7_paths = save_scope_summary_tables(scope7, "scope7", overwrite=False)
```

```python
# --- custom table exports (for polished report tables) ---
save_table(gmv_table_final, "task23_gmv_final", overwrite=True, index=False, float_format="%.5f")
save_table(tan_table_final, "task23_tan_final", overwrite=True, index=False, float_format="%.5f")
```

## Suggested naming pattern

- Figures: `taskX_short_description`
- Tables: `taskX_table_name`
- Scope sensitivity: append `_with_coal` or `_drop_coal`

Examples:
- `task3_with_coal_vs_drop_limits.png`
- `task6_proxy3_vs_ff3.csv`
- `task7_is_oos_tests.tex`
