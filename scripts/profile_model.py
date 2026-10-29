"""
Benchmark script to profile RAM usage and latency per image and per text for SigLIP 2.
Uses native Windows API via ctypes (zero extra dependencies).
"""

import ctypes
from ctypes import wintypes
import gc
import os
from pathlib import Path
import sys
import time

SIGLIP_DIR = Path(__file__).resolve().parent.parent
if str(SIGLIP_DIR) not in sys.path:
    sys.path.insert(0, str(SIGLIP_DIR))

from PIL import Image, ImageDraw
import torch

from app.model import SigLIPService


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_ram_mb() -> float:
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    if fn(handle, ctypes.byref(pmc), ctypes.sizeof(pmc)):
        return pmc.WorkingSetSize / (1024 * 1024)
    return 0.0


def benchmark():
    print("=" * 60)
    print("SigLIP 2 Local Profiling (RAM & Latency)")
    print("=" * 60)

    gc.collect()
    ram_init = get_process_ram_mb()
    print(f"Initial Process RAM: {ram_init:.2f} MB")

    t_load_start = time.perf_counter()
    service = SigLIPService(
        model_id="google/siglip2-base-patch16-224",
        torch_threads=2,
        load_now=True,
    )
    t_load = time.perf_counter() - t_load_start
    ram_loaded = get_process_ram_mb()
    model_ram_delta = ram_loaded - ram_init
    print(f"Model Load Time:    {t_load:.2f} s")
    print(f"RAM after load:     {ram_loaded:.2f} MB (Delta: +{model_ram_delta:.2f} MB)")

    # Prepare inputs
    img = Image.new("RGB", (224, 224), color="lightblue")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 170, 170], fill="blue")

    sample_text = "dirty washroom toilet bowl needing deep scrub cleaning"
    sample_texts_3 = [
        "dirty washroom toilet bowl needing deep scrub cleaning",
        "broken door lock handle needing handyman fix",
        "air conditioning cooling ceiling vent leaking water",
    ]

    # Warmup
    _ = service.encode_texts([sample_text])
    _ = service.encode_images([img])

    # Benchmark text latency (single text)
    n_runs = 10
    t_text_runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = service.encode_texts([sample_text])
        t_text_runs.append(time.perf_counter() - t0)

    avg_text_ms = (sum(t_text_runs) / len(t_text_runs)) * 1000.0
    min_text_ms = min(t_text_runs) * 1000.0

    # Benchmark image latency (single image)
    t_img_runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = service.encode_images([img])
        t_img_runs.append(time.perf_counter() - t0)

    avg_img_ms = (sum(t_img_runs) / len(t_img_runs)) * 1000.0
    min_img_ms = min(t_img_runs) * 1000.0

    # Benchmark rank latency (1 image + 3 candidates)
    candidates = [{"id": i, "text": t} for i, t in enumerate(sample_texts_3)]
    # Clear cache first to measure cold rank
    service.lru_cache.clear()
    t0 = time.perf_counter()
    _ = service.rank(img, candidates)
    cold_rank_ms = (time.perf_counter() - t0) * 1000.0

    # Warm rank (cache hit)
    t_warm_runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = service.rank(img, candidates)
        t_warm_runs.append(time.perf_counter() - t0)
    warm_rank_ms = (sum(t_warm_runs) / len(t_warm_runs)) * 1000.0

    ram_final = get_process_ram_mb()

    print("-" * 60)
    print("MEASUREMENT RESULTS (Local Windows Development Machine):")
    print(f"  Model ID:              {service.model_id}")
    print(f"  Embedding Dimension:   {service.embedding_dim}")
    print(f"  PyTorch Threads:       2")
    print(f"  Model Memory Overhead: ~{model_ram_delta:.1f} MB (Total process: ~{ram_final:.1f} MB)")
    print(f"  Latency per text:      {avg_text_ms:.2f} ms avg (min: {min_text_ms:.2f} ms)")
    print(f"  Latency per image:     {avg_img_ms:.2f} ms avg (min: {min_img_ms:.2f} ms)")
    print(f"  Cold Rank (1 img + 3 cand): {cold_rank_ms:.2f} ms")
    print(f"  Warm Rank (cached texts):   {warm_rank_ms:.2f} ms avg")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
