"""Run the pan-gan RL loop on Modal.

Why Modal at all: Prime's Hosted Training image bundles verifiers 0.1.15.dev400, which predates
the `TaskData`/`Task` split this environment is built on, and that image is not selectable
(`--image-tag` is silently ignored on managed runs — an invalid tag produced the same image).
See docs/PRIME_SUPPORT_REQUEST.md. Modal gives us the one thing that was missing: control of the
container, so we pin prime-rl and let it bring the verifiers it was built against.

    modal run modal_app/train.py                    # smoke: 2 steps, ~$2 of Pangram
    modal run modal_app/train.py --config rl,eval   # the real run: 20 steps, ~$43
    modal run modal_app/train.py --check-only       # prove the image, spend nothing
    modal run modal_app/train.py::fetch --limit 10  # read scored rollouts back

Several configs may be composed comma-separated; prime-rl deep-merges `@` files left to right.

The directory is `modal_app/`, not `modal/`, so it cannot shadow the `modal` package on import.
"""

import json
import os
import shlex
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parent.parent

# prime-rl 0.7.0. Pinned to a commit, not a branch: this is the version whose config schema every
# TOML in configs/ was validated against, and whose vendored verifiers (deps/verifiers b13ba60,
# 0.2.2.dev17) actually has `vf.TaskData`.
PRIME_RL_COMMIT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
PRIME_RL_DIR = "/opt/prime-rl"
REPO_DIR = "/root/pan-gan"
UV = "/root/.local/bin/uv"

image = (
    # devel, not runtime: prime-rl compiles CUDA extensions during sync.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git {PRIME_RL_DIR}",
        f"cd {PRIME_RL_DIR} && git checkout {PRIME_RL_COMMIT}",
        # prime-rl's .gitmodules points at git@github.com: for verifiers and renderers, which
        # needs an SSH key and a known_hosts entry the build container has neither of ("Host key
        # verification failed"). pydantic-config uses an HTTPS URL and clones fine, so only two
        # of the three fail and the cause is easy to misread.
        #
        # Rewrite .gitmodules and `submodule sync` rather than `git config insteadOf`: the
        # insteadOf rewrite did NOT apply to the submodule clones here (they still resolved the
        # SSH URL and failed identically), so fix the URLs at the source where nothing can
        # scope around them. All three repos are public.
        f"cd {PRIME_RL_DIR} && sed -i 's|git@github.com:|https://github.com/|g' .gitmodules",
        f"cd {PRIME_RL_DIR} && git submodule sync -- deps/verifiers deps/renderers deps/pydantic-config",
        # Only the submodules prime-rl's README lists for an RL run. research-environments is a
        # large checkout we never load, so it is deliberately omitted.
        # No --depth 1: submodules are pinned to exact SHAs that need not be a branch tip, and a
        # shallow fetch of a non-tip commit fails with "reference is not a tree".
        f"cd {PRIME_RL_DIR} && git submodule update --init "
        "-- deps/verifiers deps/renderers deps/pydantic-config",
        # The whole point of this image: verifiers must be new enough to have vf.TaskData, which
        # is exactly what Prime's hosted image lacked. Fail the build here, not mid-run.
        f"cd {PRIME_RL_DIR} && grep -q 'TaskData' deps/verifiers/verifiers/v1/__init__.py",
        f"cd {PRIME_RL_DIR} && {UV} sync --all-extras",
        gpu="A10G",  # some extras probe for a device at build time
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "PYTHONUNBUFFERED": "1",
            # Our taskset is imported by the env-server; keep tokenizers quiet under fork.
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Mounted at runtime rather than baked, so editing the environment or a config does not
    # rebuild the (multi-GB) image layer above.
    .add_local_dir(
        LOCAL_REPO,
        REPO_DIR,
        ignore=["~*", ".venv", ".git", "outputs", "__pycache__", "*.pyc", ".env"],
    )
)

app = modal.App("pan-gan")

# Persist model weights and run outputs across invocations. Without the HF cache a 0.8B download
# is repeated on every cold start.
cache = modal.Volume.from_name("pan-gan-cache", create_if_missing=True)
outputs = modal.Volume.from_name("pan-gan-outputs", create_if_missing=True)

# PANGRAM_API_KEY is required. WANDB_API_KEY is required by the [wandb] block in the configs.
# PRIME_API_KEY is only read by the eval overlay's craft judge.
secrets = [modal.Secret.from_name("pan-gan", required_keys=["PANGRAM_API_KEY"])]


@app.function(
    image=image,
    gpu="A100-40GB:2",  # prime-rl needs >= 2 (1 trainer + 1 inference); there is no 1-GPU RL loop
    volumes={"/cache": cache, "/outputs": outputs},
    secrets=secrets,
    timeout=24 * 60 * 60,
    retries=0,  # never silently repeat a run that already spent Pangram credit
)
def train(config: str = "rl-debug", extra_args: list[str] | None = None) -> str:
    import subprocess

    run = lambda cmd: subprocess.run(cmd, cwd=PRIME_RL_DIR, shell=True, check=True)

    # Install our environment package into prime-rl's venv. This must happen AFTER `uv sync`,
    # because sync prunes anything absent from the lockfile -- which is also why every launch
    # below uses `--no-sync`. prime-rl documents the same escape hatch for flash-attn.
    run(f"{UV} pip install -e {REPO_DIR}/environments/pangram_creative_writing")
    run(f"{UV} run --no-sync python -c 'import pangram_creative_writing; print(\"env import OK\")'")

    # `config` may name several TOMLs, comma-separated: prime-rl deep-merges `@` files left to
    # right, which is how the eval overlay composes onto a base run ("rl,eval").
    names = [c.strip() for c in config.split(",") if c.strip()]
    files = " ".join(f"@ {REPO_DIR}/configs/{n}.toml" for n in names)
    out = "/outputs/" + "-".join(names)
    cmd = (
        f"{UV} run --no-sync rl {files} "
        f"--output-dir {out} --clean-output-dir " + " ".join(extra_args or [])
    )
    print(f"launching: {cmd}", flush=True)
    try:
        subprocess.run(cmd, cwd=PRIME_RL_DIR, shell=True, check=True)
    finally:
        # Commit even when the run raises. prime-rl appends every finished rollout to
        # rollouts/step_*/train/all/traces.jsonl as it completes, so a crashed run still leaves
        # behind the rollouts it already paid Pangram for. Committing only on success would
        # discard exactly the traces worth reading after a failure.
        outputs.commit()
    return out


@app.function(image=image, gpu="A100-40GB:2", volumes={"/cache": cache}, secrets=secrets, timeout=3600)
def check() -> str:
    """Prove the image before spending anything: GPUs visible, prime-rl importable, our taskset
    loadable, and the config resolvable. Costs GPU minutes and zero Pangram credit."""
    import subprocess

    subprocess.run("nvidia-smi", shell=True, check=True)
    subprocess.run(
        f"{UV} pip install -e {REPO_DIR}/environments/pangram_creative_writing",
        cwd=PRIME_RL_DIR, shell=True, check=True,
    )
    subprocess.run(
        f"{UV} run --no-sync python -c "
        "'import verifiers, verifiers.v1 as vf, pangram_creative_writing as p; "
        'print("verifiers", verifiers.__version__, "TaskData", hasattr(vf, "TaskData")); '
        "print(\"taskset\", p.__all__)'",
        cwd=PRIME_RL_DIR, shell=True, check=True,
    )
    # --dry-run resolves and validates the full config, then exits without launching.
    subprocess.run(
        f"{UV} run --no-sync rl @ {REPO_DIR}/configs/rl-debug.toml "
        "--dry-run --output-dir /tmp/dry",
        cwd=PRIME_RL_DIR, shell=True, check=True,
    )
    return "image OK: GPUs visible, verifiers has TaskData, taskset loads, config validates"


@app.function(
    image=image,
    gpu="A100-40GB",  # inference only, no trainer; 4B in bf16 is ~8GB
    volumes={"/cache": cache, "/outputs": outputs},
    secrets=secrets,
    timeout=4 * 60 * 60,
    retries=0,
)
def calibrate(
    model: str = "Qwen/Qwen3.5-4B-Base",
    n: int = 32,
    r: int = 1,
    temperature: float = 1.0,
    directive: str = "",
    name: str = "base",
) -> str:
    """Run the calibration gate against a model Prime Inference does not host.

    Why this exists: `scripts/calibrate.sh` talks to Prime Inference, and no `-Base` checkpoint
    is listed there. The hypothesis worth testing is that what Pangram detects is the *instruct*
    register — 4B-Instruct scored 0/80 below 0.9 across four prompt/temperature settings — so the
    one model that could falsify it is exactly the one we have to serve ourselves.

    Serves vLLM on localhost and points the eval CLI at it with --client.base-url. A 2-rollout
    smoke runs first: a chat-template or thinking-mode failure gates every rollout on word_count
    and would otherwise burn the whole budget looking like a model that cannot write.
    """
    import subprocess
    import time
    import urllib.request

    env = {**os.environ, "LOCAL_API_KEY": "dummy"}  # vLLM ignores it; the client requires one

    # Same install-after-sync dance as train(): the image's `uv sync` prunes anything outside
    # prime-rl's lockfile, so our taskset has to go in afterwards and every launch uses
    # --no-sync. Without this the eval CLI reports "taskset 'pangram-creative-writing' not
    # found", which reads like a registry problem rather than a missing install.
    subprocess.run(
        f"{UV} pip install -e {REPO_DIR}/environments/pangram_creative_writing",
        cwd=PRIME_RL_DIR, shell=True, check=True, env=env,
    )

    server = subprocess.Popen(
        f"{UV} run --no-sync python -m vllm.entrypoints.openai.api_server "
        f"--model {model} --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85",
        cwd=PRIME_RL_DIR, shell=True, env=env,
    )
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited early with code {server.returncode}")
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=5)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM did not become healthy in 15 minutes")
        print("vLLM healthy", flush=True)

        def run_eval(num: int, reps: int, out: str) -> str:
            # verifiers 0.2.2.dev17 flag shape, NOT scripts/calibrate.sh's. The repo's own
            # .venv pins verifiers 0.2.1, where the taskset sits at the top level
            # (`--taskset.id`) and the harness at `--harness.id`. This image carries the
            # submodule prime-rl pins, 0.2.2.dev17, where the taskset moved onto the env
            # (`--env.taskset.id`) and the harness hangs off a named agent role
            # (`--env.agent.harness.id`). Passing the 0.2.1 shape here fails validation with
            # "the taskset lives on the env now". Same split documented in configs/rl.toml.
            cmd = (
                f"{UV} run --no-sync eval pangram-creative-writing "
                f"--env.taskset.split train "
                f"--env.taskset.style-directive {shlex.quote(directive)} "
                f"--env.agent.harness.id null --env.agent.harness.runtime.type subprocess "
                f"-m {model} -n {num} -r {reps} --shuffle --no-push -c 8 "
                f"--sampling.max-tokens 3072 --sampling.temperature {temperature} "
                f"--sampling.reasoning-effort none "
                f"--client.base-url http://localhost:8000/v1 --client.api-key-var LOCAL_API_KEY "
                f"-o {out}"
            )
            print(f"launching: {cmd}", flush=True)
            subprocess.run(cmd, cwd=PRIME_RL_DIR, shell=True, check=True, env=env)
            return f"{out}/traces.jsonl"

        # Written to /outputs, not /tmp, so a failed smoke is still inspectable after the
        # container dies -- the first version aborted on a reader bug with the evidence gone.
        smoke = run_eval(2, 1, f"/outputs/smoke-{name}")
        outputs.commit()
        rows = iter_traces(smoke)
        print(f"smoke traces: {len(rows)}", flush=True)
        stories = []
        for row in rows:
            info = row.get("info") or {}
            print(f"  words={info.get('word_count')} gate={info.get('gate')} "
                  f"ai={(row.get('metrics') or {}).get('ai_score')}", flush=True)
            print(f"  story[:200]: {(info.get('story') or '')[:200]!r}", flush=True)
            stories.append(info.get("story") or "")
        # Trust the text itself rather than any one field name.
        if not any(len(s.split()) > 50 for s in stories):
            raise RuntimeError(
                f"smoke produced no story text for {model} (word counts "
                f"{[len(s.split()) for s in stories]}). Chat template or thinking mode is wrong; "
                f"aborting before spending the full budget. Traces kept at {smoke}."
            )

        out = f"/outputs/calibration-{name}"
        traces = run_eval(n, r, out)
        summarize(traces)
        return out
    finally:
        server.terminate()
        outputs.commit()


def iter_traces(path: str) -> list[dict]:
    """Trace records from a traces.jsonl, whichever schema wrote it.

    verifiers 0.2.1 (the repo venv, and what scripts/calibrate.sh talks to) writes one flat
    trace per line. 0.2.2.dev17 (this image) writes one EPISODE per line — keys are
    `env/errors/id/ok/traces` — with the real records nested under `traces`. Reading the outer
    object on the newer schema yields `info: {}` and `metrics: {}` for every rollout, which
    looks exactly like a model that produced nothing, while the progress display shows real
    rewards. Handle both rather than depending on which venv is in front."""
    out = []
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        out.extend(row["traces"] if isinstance(row.get("traces"), list) else [row])
    return out


def summarize(traces: str) -> None:
    """The calibration gate's decision rule, applied to a traces file."""
    rows = iter_traces(traces)
    scored = [
        s for row in rows
        if (s := (row.get("metrics") or {}).get("ai_score")) is not None
    ]
    gated = len(rows) - len(scored)
    print(f"\nrollouts {len(rows)}   scored {len(scored)}   gated {gated}")
    if not scored:
        print("every rollout gated -- no distribution to read")
        return
    scored.sort()
    q = lambda p: scored[min(len(scored) - 1, int(p * len(scored)))]
    print(f"min {scored[0]:.6f}  p25 {q(.25):.6f}  p50 {q(.5):.6f}  "
          f"p75 {q(.75):.6f}  max {scored[-1]:.6f}   range {scored[-1] - scored[0]:.6f}")
    for thr in (0.5, 0.9):
        c = sum(s < thr for s in scored)
        print(f"below {thr}: {c}/{len(scored)} = {100 * c / len(scored):.1f}%")
    rate = sum(s < 0.9 for s in scored) / len(scored)
    print("\nVIABLE - there is signal to shape" if rate >= 0.05 else
          "\nNOT VIABLE at this setting - reward is effectively constant")


@app.local_entrypoint()
def main(config: str = "rl-debug", check_only: bool = False) -> None:
    if check_only:
        print(check.remote())
        return
    print(f"output dir: {train.remote(config=config)}")


@app.function(image=image, volumes={"/outputs": outputs}, timeout=900)
def fetch(config: str = "rl-debug", limit: int = 5) -> list[dict]:
    """Pull scored rollouts back off the volume — story text, detector verdict, gate reason.

    prime-rl writes these to rollouts/step_*/train/all/traces.jsonl, but that lives on a Modal
    volume, so `modal run modal_app/train.py::fetch` is how you read them without a GPU.
    """
    import json
    from pathlib import Path

    # A reader container sees another container's writes only after reload().
    outputs.reload()

    # Search from the run root, not from rollouts/: prime-rl nests its actual run under
    # <output_dir>/run_default/, so traces live at
    # <output_dir>/run_default/rollouts/step_N/train/{all,effective}/traces.jsonl.
    # The bare <output_dir>/rollouts/step_N/rank_0.bin is the trainer transport, not records.
    root = Path(f"/outputs/{'-'.join(c.strip() for c in config.split(',') if c.strip())}")
    found = sorted(root.rglob("*traces.jsonl"))
    print(f"trace files under {root}: {len(found)}")
    for f in found:
        print(f"  {f}  ({sum(1 for _ in f.open())} rollouts)")

    rows = []
    for p in sorted(root.rglob("train/all/traces.jsonl")):
        for line in p.open():
            t = json.loads(line)
            info, metrics = t.get("info", {}), t.get("metrics", {})
            rows.append(
                {
                    "step": p.parent.parent.parent.name,
                    "reward": (t.get("rewards") or {}).get("humanness"),
                    "ai_score": metrics.get("ai_score"),
                    "coherence": info.get("coherence"),
                    "gate": info.get("gate"),
                    "words": info.get("word_count"),
                    "story": (info.get("story") or "")[:400],
                }
            )
    rows.sort(key=lambda r: (r["ai_score"] is None, r["ai_score"] or 0))
    scored = [r for r in rows if r["ai_score"] is not None]
    print(f"\nrollouts: {len(rows)}  scored: {len(scored)}  gated: {len(rows) - len(scored)}")
    if scored:
        print(f"ai_score  min {min(r['ai_score'] for r in scored):.4f}  "
              f"max {max(r['ai_score'] for r in scored):.4f}")
        print(f"escaped (<0.9): {sum(r['ai_score'] < 0.9 for r in scored)}/{len(scored)}")
    print("\n--- lowest ai_score first (the escapes) ---")
    for r in rows[:limit]:
        print(f"\n[{r['step']}] ai={r['ai_score']} reward={r['reward']} "
              f"coh={r['coherence']} gate={r['gate']} words={r['words']}")
        print(f"  {r['story'][:220]!r}")
    return rows[:limit]
