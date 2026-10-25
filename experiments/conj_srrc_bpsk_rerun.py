"""Replace GMSK with SRRC BPSK in the stored conjugate robustness (ROC) results.

    python experiments/conj_srrc_bpsk_rerun.py [--workers 4]

With the GMSK generator fixed (gmsk_rerun.py), GMSK's conjugate features at 2 f_CFO +- f_sym / 2 are detected at
almost every SNR tested, so the conjugate test now uses SRRC BPSK (roll-off 0.1, span 21, as the SRRC QPSK) at
alpha = 2 f_CFO + f_sym instead, as revision_runs.roc_job does. Each conjugate file keeps its BPSK and MSK rows; its
third row is recomputed for SRRC BPSK with the same estimator, settings and procedure, and its own seed. The files
before the change are copied once to data/revision/superseded_conj_gmsk/roc. Resumable: files whose third signal is
already srrc_bpsk are skipped.
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
BACKUP = ROOT / "data" / "revision" / "superseded_conj_gmsk" / "roc"


def replace(path):
    import revision_runs as rr
    from scf_test.psk import create_srrc_bpsk_signal
    import scf_test.tests as test

    data = dict(np.load(path, allow_pickle=True))
    signals = [str(s) for s in data["signals"]]
    if signals[2] == "srrc_bpsk":
        return path.name, "skipped", 0.0
    start = time.perf_counter()
    cycle_frequency = 2 * rr.CFO + 1 / rr.SAMPLES_PER_SYMBOL
    est, Np, N, snr, trials = str(data["est"]), int(data["Np"]), int(data["N"]), float(data["snr"]), int(data["trials"])
    func, kwargs = rr.get_estimator(est, Np)
    rng = np.random.default_rng([rr.BASE_SEED, zlib.crc32((str(data["name"]) + "_srrc_bpsk").encode())])

    row, selection = np.zeros((2, trials)), None
    for j in range(2 * trials):
        clean = test.add_cfo(create_srrc_bpsk_signal(N, rr.SAMPLES_PER_SYMBOL, 21, 0.1, rng=rng), rr.CFO)
        present = j % 2 == 0
        noisy = test.add_awgn_snr(clean, desired_snr=snr, rng=rng)
        signal = noisy[:N] if present else (noisy - clean)[:N]
        coherence, _, alpha = func(signal, Np=Np, conjugate=True, coherence=True, **kwargs)
        if selection is None:
            selection = np.flatnonzero(np.abs(alpha - cycle_frequency) <= test.get_alpha_resolution(alpha) / 2)
        row[0 if present else 1, j // 2] = np.max(np.abs(coherence.ravel()[selection]))

    data["statistic"] = data["statistic"].copy()
    data["statistic"][2] = row
    data["signals"] = np.array(signals[:2] + ["srrc_bpsk"])
    cycle_freqs = np.array(data["cycle_freqs"], dtype=float)
    cycle_freqs[2] = cycle_frequency
    data["cycle_freqs"] = cycle_freqs
    data["srrc_bpsk_elapsed_seconds"] = np.array(time.perf_counter() - start)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--match", default="", help="only files whose name contains this text")
    args = parser.parse_args()

    BACKUP.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in ROC.glob("roc_*_c.npz") if args.match in p.name)
    for path in paths:
        if not (BACKUP / path.name).exists() and str(np.load(path)["signals"][2]) != "srrc_bpsk":
            shutil.copy2(path, BACKUP / path.name)
    # Cheapest first: the SSCA and FAM finish in minutes; the 2D-FFT jobs take hours
    paths.sort(key=lambda p: float(np.load(p)["elapsed_seconds"]))
    print(f"{len(paths)} conjugate ROC files, backups in {BACKUP}", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=low_priority) as pool:
        futures = [pool.submit(replace, p) for p in paths]
        for n, future in enumerate(as_completed(futures), 1):
            name, status, seconds = future.result()
            print(f"[{n}/{len(paths)}] {name} {status} ({seconds / 60:.1f} min)", flush=True)
