"""Sincronización y contexto de dispositivo sin inicializar CUDA al importar."""

import torch


def synchronize_cuda() -> None:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def device_context() -> dict[str, str | int | float | None]:
    if not torch.cuda.is_initialized():
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "gpu_allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "gpu_reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "gpu_free_mib": free / 1024**2,
        "gpu_total_mib": total / 1024**2,
        "gpu_peak_scope": "process_since_last_reset",
    }
