# machine_environments.py
from __future__ import annotations
import os
import sys
import platform
import subprocess
from pathlib import Path

# ─────────────────────────────── Hugging Face cache (no cluster names) ──────────────────────────────
def set_hf_cache(root: Path, overwrite: bool = False) -> Path:
    """
    Configure Hugging Face cache dirs to `root` (no hardcoded cluster paths).
    If overwrite=False, only set variables that are currently unset.
    Returns the cache root used.
    """
    root = Path(root).expanduser().resolve()
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)

    def _set(k: str, v: str):
        if overwrite or not os.environ.get(k):
            os.environ[k] = v

    _set("HF_HOME", str(root))
    _set("HF_DATASETS_CACHE", str(root / "datasets"))
    _set("HF_METRICS_CACHE", str(root / "metrics"))
    return root

def maybe_set_cache_from_env() -> Path | None:
    """
    Respect user-provided env vars only. No machine-specific heuristics.
    Priority:
      1) HF_HOME (already set by user)  -> leave as-is
      2) HF_CACHE_ROOT                 -> use as root
      3) SCRATCH                       -> use $SCRATCH/hf_cache
      4) Else                          -> do nothing (use HF defaults)
    """
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])

    if os.environ.get("HF_CACHE_ROOT"):
        return set_hf_cache(Path(os.environ["HF_CACHE_ROOT"]))

    if os.environ.get("SCRATCH"):
        return set_hf_cache(Path(os.environ["SCRATCH"]) / "hf_cache")

    return None

# Keep tokenizers tame; safe default, don’t override if already set.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─────────────────────────────── CUDA library path (portable) ──────────────────────────────
def maybe_extend_library_path() -> None:
    """
    Adds $CONDA_PREFIX/lib to the relevant library search path if available.
    - Linux:   LD_LIBRARY_PATH
    - macOS:   DYLD_LIBRARY_PATH
    - Windows: not modified
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_path = str(Path(conda_prefix) / "lib")

    system = platform.system()
    if system == "Darwin":
        var = "DYLD_LIBRARY_PATH"
    elif system == "Linux":
        var = "LD_LIBRARY_PATH"
    else:
        return  # Windows: typically not needed here

    existing = os.environ.get(var, "")
    parts = [lib_path] + ([p for p in existing.split(os.pathsep) if p] if existing else [])
    deduped = []
    seen = set()
    for p in parts:
        if p not in seen:
            deduped.append(p); seen.add(p)
    os.environ[var] = os.pathsep.join(deduped)

maybe_set_cache_from_env()
maybe_extend_library_path()

# ─────────────────────────────── Display Env Info ──────────────────────────────
def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            out = subprocess.check_output(["lscpu"], text=True)
            for line in out.splitlines():
                if "Model name" in line:
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif platform.system() == "Windows":
            return platform.processor() or "n/a"
    except Exception:
        pass
    return "n/a"

def print_env_info() -> None:
    import torch  # lazy import
    try:
        import transformers
        trf_ver = transformers.__version__
    except Exception:
        trf_ver = "n/a"

    print("\n\nDisplaying system, library, and hardware info.")
    print(sys.version)
    print("🐍 python       :", platform.python_version())
    print("🧠 transformers :", trf_ver)
    print("🧮 torch        :", getattr(torch, "__version__", "n/a"))

    cuda_avail = torch.cuda.is_available()
    print("🖥️  CUDA avail  :", cuda_avail)
    print("   └─ torch CUDA:", getattr(torch.version, "cuda", "n/a"))
    if cuda_avail:
        try:
            count = torch.cuda.device_count()
            print("   └─ # devices :", count)
            for i in range(count):
                print(f"   └─ device[{i}]:", torch.cuda.get_device_name(i))
        except Exception as e:
            print("   └─ CUDA query error:", repr(e))

    print("⚙️  CPU model   :", _cpu_model())
    print("-" * 60, flush=True)

if __name__ == "__main__":
    print_env_info()
