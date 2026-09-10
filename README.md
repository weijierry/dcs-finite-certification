# DCS Finite-State Certification Protocol — Reproduction Package

Reproduction archive for the preprint:

**A Typed Certification Protocol for Finite-State Causal Abstractions** (The Finite Core of the DCS Research Program), Rongjie Wei, 2026. Preprint, not peer-reviewed.

This repository reproduces the finite-model numerical tables of the preprint. All numerical results are executable certificates within specified finite models; they are not empirical confirmation in natural systems.

Status: verified on 2026-09-10 in a clean copy. All 42 tests pass; all four model groups run and reproduce the frozen baselines (three groups exactly, one within the project's own 1e-12 numeric tolerance).

## Repository structure

| Path | Content |
|---|---|
| `models/cpir_exact/` | Exact positive example and macro-kernel certificate (Theorem 1 / Corollary 1 gate) |
| `models/cpir_approximate/` | T2 numerical execution certificates: contractive and linear bounds |
| `models/selective_compression_validation/` | Non-reversible stress test: alignment vs joint admission |
| `models/cross_scale_interface/` | Joint admission prototype over all 15 partitions of a 5-state, 3-action model |
| `tests/` | Regression tests (42 cases: 12 + 9 + 10 + 11) |
| `expected_results/` | Frozen baselines |
| `复现验证记录-2026-09-10.md` | Verified run record (environment, command, comparison) |

## Environment (verified 2026-09-10)

- Python 3.14.7 (CPython, Windows amd64)
- numpy 2.5.3 (required by `selective_compression_validation` only; the other three groups use the standard library only)

```bash
pip install numpy
```

## Run instructions (clean-directory protocol)

1. Clone or unzip this repository into a fresh directory.
2. Run the model scripts from the repository root:

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

4. Compare generated outputs against `expected_results/` (see `README-reproduce.md` for the exact comparison command).

Note: `selective_compression_validation` may differ from the frozen baseline in the last 1e-15 digit on a different BLAS/LAPACK build; its regression test compares with a 1e-12 tolerance by design.

## Determinism and test accounting

- `cpir_exact`, `cpir_approximate`, `cross_scale_interface`: deterministic pure-Python computations.
- `selective_compression_validation`: deterministic given the numpy backend; eigenvalue/alignment numerics may vary in the last 1e-15 digit across BLAS/LAPACK builds (project tolerance 1e-12).
- Verified run 2026-09-10: 42/42 tests pass (cpir_exact 12, cpir_approximate 9, selective 10, cross_scale 11).

## License

CC BY 4.0 (see `LICENSE`). The paper and this reproduction package are distributed under CC BY 4.0; third-party dependencies retain their own licenses.

## Citation

See `CITATION.cff`, or cite:

> Rongjie Wei. (2026). A Typed Certification Protocol for Finite-State Causal Abstractions (version 1.0.0). https://doi.org/10.6084/m9.figshare.33519052

## Archived copies

- Figshare (published, DOI active): https://doi.org/10.6084/m9.figshare.33519052
- Mendeley Data (under review): https://data.mendeley.com/drafts/2h2m99gm5h
- Harvard Dataverse (under review): https://doi.org/10.7910/DVN/TT3HEA
- OSF Preprints / MetaArXiv (pending moderation): https://osf.io/preprints/metaarxiv/g4zrx_v1
- OSF Preprints / PsyArXiv (pending moderation): https://osf.io/preprints/psyarxiv/rpyze_v1
- SSRN (under review): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7443099
- Academia.edu (published): https://www.academia.edu/175363192/
- ai.viXra.org (submitted, ref 18267497): awaiting screening (Mathematics - General Mathematics)
- Main PDF (this repository): `DCS-finite-certification-main.pdf`
