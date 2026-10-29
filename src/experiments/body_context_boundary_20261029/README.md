# Frozen reference modules and CPU contract tests

Byte-identical modules from `gr_body_context_boundary_20261029_v1`:
the original/virtual depth-3 candidate contract, GainRatio/BIN acceptance,
body-context eligibility, target-label alignment, outcome bounds, and paired
statistics. The exact-cache utilities and boundary adapter are included as
test dependencies. The publication manifest records their hashes.

Run only the CPU contract tests from this directory:

```bash
python -m unittest -v test_contracts test_body_context_prototype
```

Python 3.10+ is required. `paired_statistics.py` additionally requires NumPy.
Importing the boundary module for its cache-key contract does not load a model.
Calling its separate prediction function does require the original local model
assets, PyTorch/Transformers, and CUDA; the command above never calls it.
These tests are engineering checks, not new model, efficacy, utility, or WSN
measurements. Production candidate extraction loads the four original text
functions from the registered scorer source; the tests use explicit synthetic
fixtures. No GPU/API launcher, credential configuration, key, raw payload,
corpus, or model checkpoint is distributed here. This directory alone does not
claim a complete end-to-end reproduction of the hosted experiment.
