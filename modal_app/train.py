"""Run the pan-gan RL loop on Modal.

Why Modal at all: Prime's Hosted Training image bundles verifiers 0.1.15.dev400, which predates
the `TaskData`/`Task` split this environment is built on, and that image is not selectable
(`--image-tag` is silently ignored on managed runs — an invalid tag produced the same image).
See docs/PRIME_SUPPORT_REQUEST.md. Modal gives us the one thing that was missing: control of the
container, so we pin prime-rl and let it bring the verifiers it was built against.

    modal run modal_app/train.py                 # 3-step smoke, ~$1.60 of Pangram
    modal run modal_app/train.py --config rl     # the real run
    modal run modal_app/train.py --gpu H100:2 --config rl

The directory is `modal_app/`, not `modal/`, so it cannot shadow the `modal` package on import.
"""

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

    cmd = (
        f"{UV} run --no-sync rl @ {REPO_DIR}/configs/{config}.toml "
        f"--output-dir /outputs/{config} --clean-output-dir "
        + " ".join(extra_args or [])
    )
    print(f"launching: {cmd}", flush=True)
    subprocess.run(cmd, cwd=PRIME_RL_DIR, shell=True, check=True)
    outputs.commit()
    return f"/outputs/{config}"


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


@app.local_entrypoint()
def main(config: str = "rl-debug", check_only: bool = False) -> None:
    if check_only:
        print(check.remote())
        return
    print(f"output dir: {train.remote(config=config)}")
