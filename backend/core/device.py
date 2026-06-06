"""Accelerator detection and acceleration planning.

Inspects the runtime - torch CUDA availability, GPU name / VRAM / compute
capability, NVIDIA driver, and whether RAPIDS cuML is importable - and recommends
how the pipeline should run: which device to embed on, the batch size to use, and
whether GPU UMAP/HDBSCAN (cuML) is available.

Deliberately dependency-light: only the standard library plus a *lazy* torch
import, so it can be imported from the backend, from a standalone benchmark, or
from a bare GPU venv that has nothing else installed.

CLI:  python backend/core/device.py        # human-readable report
      python backend/core/device.py --json  # machine-readable
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    backend: str                          # "cuda" | "cpu"
    device: str                           # "cuda:0" | "cpu"
    gpu_name: Optional[str]
    vram_mb: Optional[int]
    compute_capability: Optional[str]
    driver_version: Optional[str]
    cuda_runtime: Optional[str]           # torch.version.cuda (the build's CUDA)
    torch_version: Optional[str]
    cuml_available: bool                  # GPU UMAP/HDBSCAN possible
    embedding_batch_size: int
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _batch_for_vram(vram_mb: Optional[int]) -> int:
    """Pick an embedding batch size from available VRAM (MiniLM is small)."""
    if not vram_mb:
        return 128
    if vram_mb >= 12000:
        return 512
    if vram_mb >= 6000:
        return 256
    if vram_mb >= 3000:
        return 128
    return 64


@functools.lru_cache(maxsize=1)
def _nvidia_smi() -> Optional[dict]:
    """GPU facts straight from the driver (works even with CPU-only torch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, driver, mem, cc = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "driver": driver, "vram_mb": int(float(mem)), "cc": cc}
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _cuml_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("cuml") is not None
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def detect() -> DeviceInfo:
    """Detect the best available accelerator and the matching acceleration plan."""
    smi = _nvidia_smi()
    torch_version = cuda_runtime = None
    cuda_ok = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_runtime = getattr(torch.version, "cuda", None)
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        pass

    if cuda_ok:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            name = torch.cuda.get_device_name(0)
            vram = int(props.total_memory / (1024 * 1024))
            cc = f"{props.major}.{props.minor}"
        except Exception:
            name = smi["name"] if smi else None
            vram = smi["vram_mb"] if smi else None
            cc = smi["cc"] if smi else None
        return DeviceInfo(
            backend="cuda", device="cuda:0", gpu_name=name, vram_mb=vram,
            compute_capability=cc, driver_version=(smi["driver"] if smi else None),
            cuda_runtime=cuda_runtime, torch_version=torch_version,
            cuml_available=_cuml_available(), embedding_batch_size=_batch_for_vram(vram),
            notes="CUDA active - embeddings run on the GPU."
            + (" cuML present: GPU UMAP/HDBSCAN enabled." if _cuml_available()
               else " cuML not installed: UMAP/HDBSCAN stay on CPU."),
        )

    # CPU path - distinguish "no GPU" from "GPU present but torch can't use it".
    if smi:
        notes = (
            f"GPU detected ({smi['name']}, {smi['vram_mb']} MB, driver {smi['driver']}) "
            f"but torch cannot use it - the installed torch is CPU-only "
            f"(version {torch_version}). Install a CUDA build to enable GPU: "
            f"pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        notes = "No CUDA GPU detected - running on CPU."
    return DeviceInfo(
        backend="cpu", device="cpu",
        gpu_name=(smi["name"] if smi else None),
        vram_mb=(smi["vram_mb"] if smi else None),
        compute_capability=(smi["cc"] if smi else None),
        driver_version=(smi["driver"] if smi else None),
        cuda_runtime=cuda_runtime, torch_version=torch_version,
        cuml_available=_cuml_available(), embedding_batch_size=64, notes=notes,
    )


def _print_report() -> None:
    info = detect()
    print("ReqCluster accelerator report")
    print("=" * 60)
    print(f"  backend            : {info.backend}")
    print(f"  device             : {info.device}")
    print(f"  gpu                : {info.gpu_name or '-'}")
    print(f"  vram (MB)          : {info.vram_mb or '-'}")
    print(f"  compute capability : {info.compute_capability or '-'}")
    print(f"  nvidia driver      : {info.driver_version or '-'}")
    print(f"  torch              : {info.torch_version or '-'}")
    print(f"  torch CUDA build   : {info.cuda_runtime or 'cpu-only'}")
    print(f"  cuML (GPU cluster) : {'yes' if info.cuml_available else 'no'}")
    print(f"  embed batch size   : {info.embedding_batch_size}")
    print(f"  note               : {info.notes}")


if __name__ == "__main__":
    import sys
    if "--json" in sys.argv:
        print(json.dumps(detect().to_dict(), indent=2))
    else:
        _print_report()
