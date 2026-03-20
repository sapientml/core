# SapientML Core Efficiency Report

This report documents several areas in the codebase where efficiency improvements could be made.

## 1. Bubble Sort Algorithm in Label Ordering (High Impact)

**File:** `sapientml_core/adaptation/generation/template_based_adaptation.py`
**Lines:** 223-240

The `_sort` method uses a bubble sort algorithm with O(n^2) time complexity to order preprocessing labels. This could be replaced with Python's built-in `sorted()` function using a topological sort approach, which would be more efficient for larger label sets.

```python
def _sort(self, preprocessing_set, label_order):
    n = len(preprocessing_set)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            combination = preprocessing_set[j + 1] + "#" + preprocessing_set[j]
            if combination in label_order:
                preprocessing_set[j], preprocessing_set[j + 1] = preprocessing_set[j + 1], preprocessing_set[j]
    return preprocessing_set
```

**Recommendation:** Replace with a more efficient sorting approach using `functools.cmp_to_key` or implement a proper topological sort.

## 2. Deprecated pandas Method Usage (Medium Impact)

**File:** `sapientml_core/meta_features.py`
**Line:** 209

The code uses `applymap()` which is deprecated in pandas 2.1.0 and will be removed in a future version. It should be replaced with `map()`.

```python
is_basic_type = sampledX.applymap(
    lambda x: isinstance(x, int) or isinstance(x, float) or isinstance(x, bool) or isinstance(x, str)
)
```

**Recommendation:** Replace `applymap` with `map` for forward compatibility and to avoid deprecation warnings.

## 3. Multiple Iterations Over Same Collection (Medium Impact)

**File:** `sapientml_core/generator.py`
**Lines:** 310-322

In the `evaluate` method, the code iterates over `candidate_scripts` multiple times with separate list comprehensions when a single pass would suffice:

```python
error_pipelines = [pipeline for pipeline in candidate_scripts if pipeline[1].score is None]
# ...
succeeded_scripts = sorted(
    [x for x in candidate_scripts if x[1].score is not None],
    key=lambda x: x[1].score,
    reverse=(not lower_is_better),
)
failed_scripts = [x for x in candidate_scripts if x[1].score is None]
```

**Recommendation:** Use a single loop to partition the scripts into succeeded and failed lists, avoiding redundant iterations.

## 4. Redundant Path Construction (Low Impact)

**File:** `sapientml_core/seeding/predictor.py`
**Lines:** 271-286

The pickle file loading code has redundant path construction patterns:

```python
if python_minor_version in [9, 10, 11]:
    base_path = Path(os.path.dirname(__file__)) / ("../models/PY3" + str(python_minor_version))
    with open(base_path / "pp_models.pkl", "rb") as f:
        pp_model = pickle.load(f)
    with open(base_path / "mp_model_1.pkl", "rb") as f1:
        with open(base_path / "mp_model_2.pkl", "rb") as f2:
            m_model = (pickle.load(f1), pickle.load(f2))
else:
    with open(Path(os.path.dirname(__file__)) / "../models/pp_models.pkl", "rb") as f:
        pp_model = pickle.load(f)
    # ... similar pattern repeated
```

**Recommendation:** Consolidate path construction and simplify the conditional logic.

## 5. Inefficient Column-wise Processing (Low Impact)

**File:** `sapientml_core/params.py`
**Lines:** 385-394

In `summarize_dataset`, meta features are generated for each column individually in a loop:

```python
for column_name in df_train.columns:
    meta_features = generate_column_meta_features(df_train[[column_name]])
```

**Recommendation:** Consider batch processing columns where possible to reduce overhead.

## Selected Fix

For this PR, I will fix **Issue #1: Bubble Sort Algorithm** as it provides the clearest algorithmic improvement from O(n^2) to O(n log n) complexity, which can have a noticeable impact when dealing with larger sets of preprocessing labels.
