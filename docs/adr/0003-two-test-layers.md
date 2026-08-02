# 0003: Two test layers: fast/offline by default, integration opt-in

**Status:** Accepted · **Date:** 2026-08-02

## Context

Two kinds of failure can reach a user here, and they need different tests.

The first is arithmetic: a letterbox undone in the wrong order, a run-length
key that ignores the bottom pixel, a Cityscapes class mapped to the wrong tile.
Synthetic fixtures catch all of it, in milliseconds, with no infrastructure.

The second is rot in things we do not control. Both default model URLs were
silently deleted upstream and started returning `401`, which killed `--detect`
and `--segment` on every cold cache while the entire test suite stayed green.
Nothing built on synthetic tensors could ever have noticed.

The temptation is one suite that does both. That fails in a specific way: the
pre-push hook (`.githooks/pre-push`) runs `uv run pytest` on every push, so any
test that reaches the network makes pushing depend on Hugging Face's mood.
The first time a push is blocked by a 30-second timeout on an unrelated commit,
the developer learns `--no-verify`, and the hook stops being a gate at all.

## Decision

Two layers, split by cost, in one repo.

The fast layer is `tests/*.py`: synthetic images, synthetic tensors, a fake
ONNX session, no model and no network. It is the default
(`addopts = "-m 'not integration'"` in `pyproject.toml` deselects everything
else), and it is what the pre-push hook and the required CI jobs run. 150 tests
in well under a second.

The heavy layer is `tests/integration/`, marked `integration` and run only when
asked for with `uv run pytest -m integration`. It downloads the real ~28 MB
weights, builds real ONNX Runtime sessions, checks real tensor shapes, and
pushes a live camera frame through the full pipelines. Missing infrastructure
**skips**, never fails.

In CI the heavy layer is a separate advisory job on a weekly cron plus manual
dispatch, with `continue-on-error: true`. It never gates a pull request.

## Consequences

The fast layer must stay infra-free forever. Not "usually offline": a single
network call in `tests/*.py` re-couples every push to someone else's uptime,
and the sub-second runtime is the only reason the hook is tolerable in the
first place. New tests that need a model or the API go in `tests/integration/`,
no exceptions.

Deselection happens after collection, so pytest still *imports* every
integration module on a default run. Those modules must keep imports cheap
(`onnxruntime` is loaded inside fixtures, not at module scope), or the fast
layer pays for the heavy one anyway.

Skip-don't-fail means a genuinely broken mirror looks the same as no internet.
That is the right trade for a suite that must run offline, but it makes the
weekly scheduled run the actual detection mechanism: read the job summary.
Nobody is going to be paged.

Advisory jobs also drift. `tests/integration/test_model_sources.py` is the only
thing in the repo that notices a dead download URL, so it earns its keep even
though it is allowed to fail. Same reasoning as the `pip-audit` job in
`.github/workflows/ci.yml`: loud where people look, never in the merge path.
