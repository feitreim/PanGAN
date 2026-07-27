# Support request — Hosted Training image ships verifiers 0.1.15.dev400

**Account:** feitreim (user `cms3dmhle004w12j2d9spoveu`) · **CLI:** 0.6.20
**Environment:** `feitreim/pangram-creative-writing@0.1.3` (PRIVATE)
**Failed runs:** `xxb3nrcfq1anhovk3vbiifkr`, `kjivjycf8x98j5dqvd0n6jio`,
`s5kduogg1mvscjdwn6ov621e`, `nq8m5s3szznn8x8z3rnd3g8m`, `ziahn5xp5pl5maabl2k4xhzm`

## Summary

Hosted Training's runtime image bundles **verifiers 0.1.15.dev400**, which predates the
`TaskData` / `Task` split. Any environment scaffolded by the current v1 CLI (`uv run init`,
verifiers 0.2.1) fails at import inside the orchestrator and every env-server process:

```
RuntimeError: verifiers 0.1.15.dev400 at /app/deps/verifiers/verifiers/__init__.py
has no verifiers.v1.TaskData
```

(That message is our own guard; the raw error is
`AttributeError: module 'verifiers.v1' has no attribute 'TaskData'. Did you mean: 'Taskset'?`)

## The core inconsistency

**Your build container and your training container disagree about verifiers.** The Hub's
integration test for this exact package passes — including `test_install_and_import`, which
imports the taskset and therefore resolves `vf.TaskData` successfully. The same wheel then
fails to import in the training runtime. One platform, one package, two incompatible runtimes.

For reference, everything else in the ecosystem is on 0.2.x:

| component | verifiers |
|---|---|
| PyPI stable | 0.2.1 (has `TaskData`) |
| prime-rl 0.7.0 (`deps/verifiers` = `b13ba60`) | 0.2.2.dev17 (has `TaskData`) |
| `uv run init` scaffold output | targets 0.2.x |
| Hub CI container | 0.2.x (import passes) |
| **Hosted Training runtime** | **0.1.15.dev400** |

`prime train init` also emits a template documenting the v1 shape
(`taskset = { id = ... }`, `harness = { id = "default", ... }`) that this runtime cannot load.

## What we ruled out

- **Not an API rename.** `TaskData` is present in 0.2.1 and in 0.2.2.dev17/25/30/36.
- **Not a partial import.** `verifiers/v1/__init__.py` binds `TaskData` before `Taskset`, so a
  half-initialised module would be missing `Taskset`, not `TaskData`.
- **Not fixable by pinning.** Declaring `verifiers>=0.2.1` in the environment's own
  `pyproject.toml` changes nothing: the install runs as
  `uv pip install --python /app/.venv/bin/python -P pangram_creative_writing <wheel>`, and `-P`
  scopes upgrades to that one package, so the vendored path-install of verifiers is left in
  place. uv reports `Installed 1 package` and the unsatisfiable constraint is silently dropped
  rather than failing loudly.
- **Not fixable by `--image-tag`.** On this managed/LoRA run type the flag appears to be
  ignored. `--image-tag latest` and `--image-tag definitely-not-a-real-tag-zzz` both produced
  the same 0.1.15.dev400 image, and the invalid tag was accepted without error rather than
  failing on an unresolvable manifest.
- **Not addressable with Prime Images.** Those are sandbox images; this environment uses
  `runtime = { type = "subprocess" }` and never creates a sandbox. The failure is in the
  prime-rl container itself.

## Requests

1. **Update the Hosted Training image** to a prime-rl build whose vendored verifiers matches
   the version the environment CLI scaffolds against (0.2.x). This affects every native v1
   environment on the platform, not just ours.
2. **Enable full fine-tuning / GPU dispatch on this account** so `--image-tag` becomes usable.
   `prime train gpus` currently returns "No GPU types available. Contact support if you need
   access." That would let us pin a current image ourselves rather than wait on a release.
3. Two smaller things worth fixing while you're in there:
   - `--image-tag` silently ignoring an invalid tag on managed runs should be a validation
     error, not a no-op — it cost us several runs to discover.
   - The Hub CI asserts `tags` exists in `[project]`, but the v1 scaffold (`uv run init`) does
     not emit it while the v0 scaffold (`prime env init`) does, so a correctly-built v1 package
     fails CI on first push.

Happy to grant access to the environment or re-run anything that would help diagnose.
