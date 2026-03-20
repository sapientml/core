# sapientml/core — Agent Memory

## Repository Overview
- **Purpose**: SapientML plugin that generates scikit-learn ML pipelines from tabular data
- **Package**: `sapientml-core`; entry-points register config, generator, datastore, preprocess
- **Python support**: `>=3.9,<3.14`; CI matrix covers 3.10, 3.11, 3.12, 3.13
- **Package manager**: uv (migrated from Poetry in `build/update-dependencies` → PR #113)
- **Build backend**: hatchling (PEP 621)

## Key Dependencies & Constraints

| Package | Constraint | Reason |
|---------|-----------|--------|
| `xgboost` | `>=1.7.6,<3.0.0` | XGBoost 3.x stores `base_score` as bracketed UBJ string `'[5.0482047E-1]'`; SHAP 0.49.1 `float()` parse fails |
| `shap` | `>=0.43,<0.52` | SHAP 0.50+ requires `numpy>=2` |
| `numpy` | `>=1.19.5,<3.0.0` (via `[tool.uv] override-dependencies`) | Overrides `sapientml`'s legacy `<2.0.0` cap (that cap was for fasttext-wheel, now replaced by langdetect). uv resolves numpy 2.0.2/2.2.6/2.4.3 per Python version. |
| `numba` | `>=0.57.1,<0.65.0` | SHAP transitive dep; 0.60.0 (Python <3.10) requires numpy<2.1; 0.64.0 (Python >=3.10) requires numpy<2.5 |
| `langdetect` | `>=1.0.9` | Replaced `fasttext-wheel` (no cp313 wheels, numpy<2 required) |
| `isort` | `<8.0` | isort 8.x changes pysen integration |
| `scikit-learn` | `>=1.6.1,<2.0` | `_validate_data` removed in 1.6; public `validate_data` used |

## SHAP / XGBoost Compatibility
- **Root cause**: XGBoost 3.x UBJ serialisation wraps `base_score` in brackets; `shap.XGBTreeModelLoader` calls `float('[5.0...]')` → `ValueError`
- **Fix**: `xgboost<3.0.0` in `pyproject.toml` → uv resolves `xgboost==2.1.4`
- **Template**: `sapientml_core/templates/other_templates/shap.py.jinja` uses `shap.TreeExplainer(model)` directly for all tree models

## Python 3.13 / numpy 2.x Support (PR #116)
- **Problem**: `numpy==1.26.4` has no cp313 wheel → source build in CI → job timeout
- **Root cause chain**: `sapientml` (PyPI ≤0.4.16) required `numpy<2.0.0` due to `fasttext-wheel` incompatibility (sapientml commit 279f99d). fasttext-wheel has been replaced by `langdetect` in sapientml-core.
- **Fix**: `[tool.uv] override-dependencies = ["numpy>=1.19.5,<3.0.0"]` in `pyproject.toml` overrides sapientml's transitive constraint.
- **Lock split**: uv now resolves three numpy versions by Python range:
  - `2.0.2` for Python <3.10 (numba 0.60.0 requires `<2.1`)
  - `2.2.6` for Python 3.10 (last numpy supporting 3.10)
  - `2.4.3` for Python ≥3.11 (has cp313 wheels; numba 0.64.0 requires `<2.5`)
- **Companion PR**: `sapientml/sapientml` PR #112 removes the `<2.0.0` cap and extends `requires-python` to `<3.14` permanently
- **CI fix**: Added `fail-fast: false`; coverage artifact names now include Python version (`py${{ver}}-${{test}}`) to avoid overwrite collisions

## Pull Request Requirements (must do before opening or updating a PR)
1. **DCO sign-off**: Every commit on the branch must carry `Signed-off-by: <name> <email>`.
   - Add to new commits: `git commit -s`
   - Retrofit all existing commits on branch: `git rebase --signoff main` (then `git push --force-with-lease`)
   - If a commit already has a duplicate sign-off, remove it with interactive rebase (`git rebase -i`) and `git commit --amend`.
2. **All tests green**: CI must pass for every Python version in the matrix (currently 3.10, 3.11, 3.12, 3.13) before requesting a review or merging.
   - Watch a run: `gh run watch <RUN_ID> --repo sapientml/core --interval 30`
   - Rerun failed jobs: `gh run rerun <RUN_ID> --repo sapientml/core --failed`

## Releases
| Version | Date | Notes |
|---------|------|-------|
| 0.7.4 | 2026-03-18 | uv migration (#113), bool dtype fix (#114) |

## CI / GitHub Actions
- Workflow: `.github/workflows/test.yml`
- `test_regressor_works_number[target_number-r2]` / `[target_number-RMSE]` exercise SHAP with generated XGBoost code

## Common Commands
```bash
# Install / sync venv
uv sync

# Run specific test
uv run pytest tests/sapientml/test_generatedcode.py::test_regressor_works_number -v

# Check shap+xgboost compat quickly
uv run python3 -c "
import xgboost as xgb, numpy as np, shap
m = xgb.XGBRegressor(n_estimators=5).fit(np.random.rand(50,4), np.random.rand(50))
print(shap.TreeExplainer(m)(np.random.rand(5,4)).values.shape)
"

# Regenerate lock (after pyproject.toml changes)
uv lock

# CI status
gh run list --repo sapientml/core --branch build/update-dependencies --limit 5
```

## Issue #84 — bool dtype ValueError in scipy stats helpers
- **File**: `sapientml_core/meta_features.py`
- **Affected functions**: `_get_ttest_pvalue`, `_get_kstest_pvalue`, `_get_pearsonr_values`
- **Root cause**: `_is_realbool_dtype()` selects `bool` dtype columns alongside numeric ones. scipy.stats internals call `numpy.finfo(numpy.bool_)` which raises `ValueError: data type <class 'numpy.bool_'> not inexact`
- **Fix**: `c = c.astype(float)` at the entry of each helper (PR #114, branch `fix/issue-84-bool-dtype-scipy-stats`)
- **Tests**: `tests/sapientml/test_meta_features.py` — 10 unit tests for bool/float/edge cases
- **Symptom**: `pytest -k test_classifier_category_binary_num_proba[target_category_binary_num-auc]` → `ValueError`

## pkl Fixtures
- `tests/fixtures/models/*.pkl`: pre-serialised model files per Python version; regenerated by `fix: re-serialize model pkl files` commit if sklearn version changes

## Commit History (build/update-dependencies)
| SHA | Description |
|-----|-------------|
| cd758db | build: migrate package manager from Poetry to uv |
| 05387d9 | fix: constrain isort<8.0 and bump pysen to 0.12.0 |
| f7bbaf2 | fix: replace removed sklearn _validate_data with public validate_data |
| 49e3a09 | fix: remove pandas FutureWarnings and deprecated API usage |
| cc65117 | fix: re-serialize model pkl files to match current sklearn versions per Python |
| 4d0cd67 | fix: pin xgboost<3.0.0 to fix SHAP 0.49.x incompatibility |
