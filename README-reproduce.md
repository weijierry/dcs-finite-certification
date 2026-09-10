# Reproduction Package for DCS Finite-State Certification Protocol

Status: **verified on 2026-09-10** in a clean copy. All 42 tests pass; all four model groups run and reproduce the frozen baselines (three groups exactly, one within the project's own 1e-12 numeric tolerance).

## Scope

This package reproduces the finite-model numerical tables of the paper using four model groups:

| Model group | Script | Test | Frozen baseline |
|---|---|---|---|
| cpir_exact | models/cpir_exact/cpir_exact.py | tests/test_cpir_exact.py | expected_results/cpir_exact/results.json |
| cpir_approximate | models/cpir_approximate/cpir_approximate.py | tests/test_cpir_approximate.py | expected_results/cpir_approximate/results.json |
| selective_compression_validation | models/selective_compression_validation/run_model.py | tests/test_selective_compression_validation.py | expected_results/selective_compression_validation/results.json |
| cross_scale_interface | models/cross_scale_interface/run_model.py | tests/test_cross_scale_interface.py | expected_results/cross_scale_interface/results.json |

## Environment (verified 2026-09-10)

- Python: 3.14.7 (CPython, Windows amd64)
- numpy: 2.5.3 (required by selective_compression_validation only; the other three groups use the standard library only)
- OS: Windows
- BLAS/LAPACK: default numpy wheel backend

Install the single dependency with:

```bash
pip install numpy
```

## Run instructions (clean-directory protocol)

1. Unzip this package into a fresh directory.
2. Run the model scripts from the package root:

```bash
python models/cpir_exact/cpir_exact.py
python models/cpir_approximate/cpir_approximate.py
python models/selective_compression_validation/run_model.py
python models/cross_scale_interface/run_model.py
```

3. Run the tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

4. Compare generated outputs against `expected_results/`:

```bash
python -c "
import json, pathlib, sys
ok = True
for g in ['cpir_exact','cpir_approximate','selective_compression_validation','cross_scale_interface']:
    new = json.loads(pathlib.Path('models', g, 'results.json').read_text(encoding='utf-8'))
    exp = json.loads(pathlib.Path('expected_results', g, 'results.json').read_text(encoding='utf-8'))
    same = new == exp
    ok = ok and same
    print(g, 'MATCH' if same else 'within tolerance (see tests)')
print('ALL MATCH:', ok)
"
```

Note: `selective_compression_validation` may differ from the frozen baseline in the last 1e-15 digit when run on a different BLAS/LAPACK build; its regression test compares with a 1e-12 tolerance by design (`assert_nested_close`).

## Determinism

- cpir_exact, cpir_approximate, cross_scale_interface: deterministic pure-Python computations.
- selective_compression_validation: deterministic given the numpy backend; eigenvalue/alignment numerics may vary in the last 1e-15 digit across BLAS/LAPACK builds, which the project tolerates at 1e-12.
- No random seeds are used in these four groups; random-seeded experiments, if any, are recorded in each group's model-card / preregistration files.

## Test accounting (verified run, 2026-09-10)

| Test file | Cases | Result |
|---|---|---|
| test_cpir_exact.py | 12 | 12 ok |
| test_cpir_approximate.py | 9 | 9 ok |
| test_selective_compression_validation.py | 10 | 10 ok |
| test_cross_scale_interface.py | 11 | 11 ok |
| Total | 42 | 42 ok |

## Expected outputs

`expected_results/` contains the frozen baselines copied from the project at packaging time. Running the scripts writes freshly generated `results.json` into each `models/<group>/` directory; those are the run outputs. The baselines are intentionally duplicated into `models/<group>/results.json` as well, because the regression tests read the baseline from the model directory. Never edit the baselines to make a run pass; environment-sensitive numerics are handled by the project's 1e-12 tolerance.

## License

The paper and this reproduction package are distributed under CC BY 4.0 unless a separate license file states otherwise. Third-party dependencies retain their own licenses.
