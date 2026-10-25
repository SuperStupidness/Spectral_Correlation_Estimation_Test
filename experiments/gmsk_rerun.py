"""Regenerate the GMSK row of every stored robustness (ROC) result after the GMSK generator fix.

    python experiments/gmsk_rerun.py [--workers 8]

scf_test.fsk.create_gmsk_signal used a Gaussian filter that was not normalized to unit gain, so its modulation
index was about 0.55 instead of 0.5. Every ROC job in data/revision/roc keeps the statistics of its three signals
separately, (signal, [present, absent], trial), so only the GMSK row has to be recomputed: same estimator, settings,
cycle frequency and procedure as revision_runs.roc_job, with the corrected generator and its own seed. The rows of
the other signals are kept. The files before the fix are copied once to data/revision/superseded_gmsk_h055/roc.
Resumable: a file already carrying gmsk_fixed is skipped.
"""
import argparse
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

ROC = ROOT / "data" / "revision" / "roc"
BACKUP = ROOT / "data" / "revision" / "superseded_gmsk_h055" / "roc"


def regenerate(path):
    import revision_runs as rr
    from scf_test.fsk import create_gmsk_signal
    import scf_test.tests as test

    data = dict(np.load(path, allow_pickle=True))
    if "gmsk_fixed" in data:
        return path.name, "skipped", 0.0
    start = time.perf_counter()
    signals = [str(s) for s in data["signals"]]
    index = signals.index("gmsk")
    cycle_frequency = float(data["cycle_freqs"][index])
    est, Np, N, snr = str(data["est"]), int(data["Np"]), int(data["N"]), float(data["snr"])
    conjugate, trials = bool(data["conj"]), int(data["trials"])
    func, kwargs = rr.get_estimator(est, Np)
    rng = np.random.default_rng([rr.BASE_SEED, zlib.crc32((str(data["name"]) + "_gmsk_h05").encode())])

    row, selection = np.zeros((2, trials)), None
    for j in range(2 * trials):
        clean = create_gmsk_signal(N, rr.SAMPLES_PER_SYMBOL, f_carrier=rr.CFO, BT=0.3, gauss_filter_span=21, rng=rng)[0]
        present = j % 2 == 0
        noisy = test.add_awgn_snr(clean, desired_snr=snr, rng=rng)
        signal = noisy[:N] if present else (noisy - clean)[:N]
        coherence, _, alpha = func(signal, Np=Np, conjugate=conjugate, coherence=True, **kwargs)
        if selection is None:
            selection = np.flatnonzero(np.abs(alpha - cycle_frequency) <= test.get_alpha_resolution(alpha) / 2)
        row[0 if present else 1, j // 2] = np.max(np.abs(coherence.ravel()[selection]))

    data["statistic"] = data["statistic"].copy()
    data["statistic"][index] = row
    data["gmsk_fixed"] = np.array(True)
    data["gmsk_elapsed_seconds"] = np.array(time.perf_counter() - start)
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as file:
        np.savez(file, **data)
    for attempt in range(10):  # a sync client can hold the file open for a moment
        try:
            partial.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(2)
    return path.name, "done", time.perf_counter() - start


def low_priority():
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # idle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--match", default="", help="only files whose name contains this text")
    args = parser.parse_args()

    BACKUP.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in ROC.glob("roc_*.npz") if args.match in p.name)
    for path in paths:
        if not (BACKUP / path.name).exists() and "gmsk_fixed" not in np.load(path).files:
            shutil.copy2(path, BACKUP / path.name)
    # Longest jobs first (2D-FFT, then Fast-SC at large Np), so the pool does not end on one long job
    cost = lambda p: float(np.load(p)["elapsed_seconds"])
    paths.sort(key=cost, reverse=True)
    print(f"{len(paths)} ROC files, backups in {BACKUP}", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=low_priority) as pool:
        futures = [pool.submit(regenerate, p) for p in paths]
        for n, future in enumerate(as_completed(futures), 1):
            name, status, seconds = future.result()
            print(f"[{n}/{len(paths)}] {name} {status} ({seconds / 60:.1f} min)", flush=True)
