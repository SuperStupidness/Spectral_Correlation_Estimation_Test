"""Views of an estimated SCD (or coherence) that stay readable where a 3D surface does not.

Every estimator returns (Sx, f, alpha) as scattered samples on its own lattice (the SSCA's diamond, the
FAM's channel pairs, ...), with Np x N or N x N points. A surface over all of them is too dense to draw
and hides the thin cycle-frequency features behind one another. Here the samples are max-pooled onto a
regular grid (the largest magnitude in each cell, so a spike on a single cycle frequency survives the
pooling) and shown as

    a heatmap over (f, alpha) in dB, with the cycle-frequency profile max_f |Sx| beside it on the same
    alpha axis, so each feature lines up with its spike in the profile; and
    a 3D view of only the significant alpha slices, like the textbook pictures of the SCD, since the SCD
    is zero away from the cycle frequencies of the signal.

Every estimator here computes its cycle frequencies on a lattice 1/N apart, so a slice is the estimates
at one point of that lattice. plot_scd_waterfall draws all the slices within a few steps of one cycle
frequency, and scd_slice_explorer writes a page that steps through them one at a time.

    from scf_test import ssca
    from scf_test.visualization.scd import plot_scd, plot_scd_waterfall, scd_slice_explorer
    Sx, f, alpha = ssca(signal, Np=64)
    plot_scd(Sx, f, alpha)
    plot_scd_waterfall(Sx, f, alpha, alpha0=0.1)
    scd_slice_explorer(Sx, f, alpha, "scd.html")  # open in a browser
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import find_peaks


def pool_max(Sx, f, alpha, n_f=256, n_alpha=1024, f_range=None, alpha_range=None):
    """Largest |Sx| in each cell of a regular (alpha, f) grid; NaN where no estimate falls.

    Returns (grid, f_centres, alpha_centres), with grid of shape (n_alpha, n_f).
    """
    values, f, alpha = np.abs(Sx).ravel(), np.ravel(f), np.ravel(alpha)
    f_range = f_range or (f.min(), f.max())
    alpha_range = alpha_range or (alpha.min(), alpha.max())
    f_edges = np.linspace(*f_range, n_f + 1)
    alpha_edges = np.linspace(*alpha_range, n_alpha + 1)
    i = np.clip(np.searchsorted(alpha_edges, alpha, side="right") - 1, 0, n_alpha - 1)
    j = np.clip(np.searchsorted(f_edges, f, side="right") - 1, 0, n_f - 1)
    inside = ((alpha >= alpha_range[0]) & (alpha <= alpha_range[1])
              & (f >= f_range[0]) & (f <= f_range[1]) & np.isfinite(values))

    grid = np.full(n_alpha * n_f, -np.inf)
    np.maximum.at(grid, i[inside] * n_f + j[inside], values[inside])
    grid[np.isinf(grid)] = np.nan
    centres = lambda edges: (edges[:-1] + edges[1:]) / 2
    return grid.reshape(n_alpha, n_f), centres(f_edges), centres(alpha_edges)


def fill_rows(grid):
    """Fill the empty cells between the first and last estimate of each alpha row with the nearest estimate.

    The estimators resolve f only to about 1/Np, so a fine grid has empty cells inside the support; the
    cells outside it (beyond the first and last estimate of a row) stay empty.
    """
    filled = grid.copy()
    columns = np.arange(grid.shape[1])
    for row in filled:
        known = np.flatnonzero(np.isfinite(row))
        if len(known) < 2:
            continue
        gaps = columns[known[0]:known[-1]]
        gaps = gaps[np.isnan(row[gaps])]
        right = np.searchsorted(known, gaps)
        nearest = np.where(gaps - known[right - 1] <= known[right] - gaps, known[right - 1], known[right])
        row[gaps] = row[nearest]
    return filled


def alpha_lattice(alpha):
    """The cycle frequencies the estimator computes, and their spacing (1/N for every estimator here)."""
    lattice = np.unique(np.round(np.ravel(alpha), 12))
    return lattice, np.median(np.diff(lattice))


def lattice_slices(Sx, f, alpha, n_f=256):
    """|Sx(f, alpha)| at every cycle frequency of the lattice, max-pooled onto n_f frequency bins.

    Returns (grid, f_centres, lattice), with grid of shape (len(lattice), n_f) and NaN where no estimate
    falls (f is resolved only to about 1/Np).
    """
    lattice, step = alpha_lattice(alpha)
    grid, f_centres, _ = pool_max(Sx, f, alpha, n_f, len(lattice),
                                  alpha_range=(lattice[0] - step / 2, lattice[-1] + step / 2))
    return grid, f_centres, lattice


def cycle_frequency_profile(Sx, f, alpha):
    """max_f |Sx(f, alpha)| at every cycle frequency of the lattice. Returns (profile, lattice)."""
    grid, _, lattice = lattice_slices(Sx, f, alpha, n_f=1)
    return grid[:, 0], lattice


def significant_cycle_frequencies(profile, alpha_centres, n_slices=6, conjugate=False, min_prominence=3):
    """Up to n_slices most prominent spikes of the cycle-frequency profile, min_prominence dB or more.

    Ranked by prominence in dB, so a spike stands out against the floor next to it rather than merely
    being high (the leakage around alpha = 0 is high but broad). For the non-conjugate SCD, which is
    symmetric in alpha, only alpha > 0 is searched and alpha = 0 (the PSD) is added.
    """
    db = 10 * np.log10(np.maximum(np.nan_to_num(profile, nan=0.0), np.finfo(float).tiny))
    db = np.maximum(db, np.nanmax(db) - 60)
    peaks, properties = find_peaks(db, distance=max(1, len(profile) // 100), prominence=min_prominence)
    keep = alpha_centres[peaks] > 0.5 * (alpha_centres[1] - alpha_centres[0]) if not conjugate else np.ones(len(peaks), bool)
    peaks, prominence = peaks[keep], properties["prominences"][keep]
    chosen = list(alpha_centres[peaks[np.argsort(prominence)[::-1][:n_slices]]])
    if not conjugate:
        chosen.append(0.0)
    return np.sort(chosen)


def scd_slice(Sx, f, alpha, alpha0, n_f=256, lattice=None):
    """|Sx(f, alpha)| at the cycle frequency of the lattice nearest alpha0, on n_f frequency bins.

    Returns (values, f_centres, alpha), keeping the bins that hold an estimate. Pass the output of
    alpha_lattice as `lattice` when taking many slices of a large estimate, to find it only once.
    """
    lattice, step = alpha_lattice(alpha) if lattice is None else lattice
    alpha0 = lattice[np.argmin(np.abs(lattice - alpha0))]
    grid, f_centres, _ = pool_max(Sx, f, alpha, n_f=n_f, n_alpha=1,
                                  alpha_range=(alpha0 - step / 2, alpha0 + step / 2))
    known = np.isfinite(grid[0])
    return grid[0][known], f_centres[known], alpha0


def plot_slices_3d(Sx, f, alpha, alphas, ax=None, n_f=256, cmap="viridis", label="|S|"):
    """The slices at the cycle frequencies `alphas`, drawn as filled curves in 3D (linear magnitude)."""
    if ax is None:
        ax = plt.figure(figsize=(7, 4.6)).add_subplot(projection="3d")
    lattice = alpha_lattice(alpha)
    slices = [scd_slice(Sx, f, alpha, a, n_f, lattice) for a in alphas]
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.85, len(slices)))
    for (values, f_slice, a), color in zip(slices, colors):
        verts = [list(zip(f_slice, np.full_like(f_slice, a), values)) + [(f_slice[-1], a, 0), (f_slice[0], a, 0)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=0.35, edgecolor="none"))
        ax.plot(f_slice, np.full_like(f_slice, a), values, color=color, linewidth=0.9)
    snapped = [a for _, _, a in slices]
    ax.set_xlim(np.min(f), np.max(f))
    ax.set_ylim(min(snapped) - 0.05, max(snapped) + 0.05)
    ax.set_zlim(0, max(np.max(values) for values, _, _ in slices) * 1.05)
    ax.set_xlabel("f")
    ax.set_ylabel(r"$\alpha$")
    ax.set_zlabel(label)
    ax.view_init(elev=28, azim=-58)
    return ax


def _to_db(values, dynamic_range):
    db = 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
    top = np.nanmax(db)
    return np.clip(db, top - dynamic_range, top), top


def plot_scd(Sx, f, alpha, conjugate=False, alphas=None, n_slices=6, db=True, dynamic_range=40, n_f=256,
             n_alpha=400, title=None, cmap="viridis"):
    """Heatmap with the cycle-frequency profile beside it, and the significant slices in 3D.

    alphas : cycle frequencies for the 3D slices (e.g. signals.expected_cycle_frequencies); by default the
        n_slices most prominent spikes of the profile (see significant_cycle_frequencies).
    n_alpha : rows of the heatmap. Keep it near the height of the map in pixels: with more rows than
        pixels, the renderer drops rows, and with them the one-row cycle-frequency features.
    db : heatmap and profile in dB (10 log10 |Sx|), clipped to dynamic_range below the peak. Use db=False
        for the coherence, whose magnitude is already on a 0 to 1 scale. The 3D slices are always linear,
        as in the textbook pictures.
    """
    grid, f_centres, alpha_centres = pool_max(Sx, f, alpha, n_f, n_alpha)
    grid = fill_rows(grid)
    profile, profile_alpha = cycle_frequency_profile(Sx, f, alpha)
    if alphas is None:
        alphas = significant_cycle_frequencies(profile, profile_alpha, n_slices, conjugate)

    fig = plt.figure(figsize=(12, 4.4))
    spec = fig.add_gridspec(1, 4, width_ratios=[1, 0.3, 0.04, 1.45], wspace=0.08)
    ax_map = fig.add_subplot(spec[0])
    ax_profile = fig.add_subplot(spec[1], sharey=ax_map)
    ax_bar = fig.add_subplot(spec[2])
    ax_3d = fig.add_subplot(spec[3], projection="3d")

    unit = "dB" if db else "magnitude"
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#e6e6e6")  # outside the estimator's support
    extent = (f_centres[0], f_centres[-1], alpha_centres[0], alpha_centres[-1])
    image = ax_map.imshow(_to_db(grid, dynamic_range)[0] if db else grid, origin="lower", aspect="auto",
                          extent=extent, cmap=colormap, interpolation="nearest")
    fig.colorbar(image, cax=ax_bar, label=f"|S| ({unit}), max-pooled")
    ax_map.set_xlabel("Frequency f")
    ax_map.set_ylabel(r"Cycle frequency $\alpha$")
    # Drawn max-pooled at twice the rows of the map: at the full 1/N spacing the line fills solid
    shown, _, shown_alpha = pool_max(profile, np.zeros_like(profile), profile_alpha, n_f=1, n_alpha=2 * n_alpha)
    shown = shown[:, 0]
    ax_profile.plot(_to_db(shown, dynamic_range)[0] if db else shown, shown_alpha, color="black", linewidth=0.7)
    ax_profile.set_xlabel(r"max$_f$ |S| " + f"({unit})")
    ax_profile.tick_params(labelleft=False)
    ax_profile.grid(True, alpha=0.3)
    ax_profile.set_ylim(alpha_centres[0], alpha_centres[-1])
    ax_profile.set_title("Cycle-frequency profile", fontsize=8)
    ax_profile.plot(np.full(len(alphas), 1), alphas, "<", color="tab:red", ms=4,
                    transform=ax_profile.get_yaxis_transform(), clip_on=False)  # the slices drawn in 3D

    plot_slices_3d(Sx, f, alpha, alphas, ax_3d, n_f, cmap)
    ax_3d.set_title(r"Slices at $\alpha$ = " + ", ".join(f"{a:.3g}" for a in alphas), fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def plot_scd_waterfall(Sx, f, alpha, alpha0, half_width=20, n_f=256, db=False, dynamic_range=40, ax=None):
    """Every slice from alpha0 - half_width to alpha0 + half_width lattice steps (1/N each), in 3D.

    Shows how a feature at alpha0 spreads over the neighbouring cycle frequencies: the cycle-frequency
    resolution and the cycle leakage of the estimator. The slice at alpha0 is drawn in black.
    """
    lattice, step = alpha_lattice(alpha)
    centre = np.argmin(np.abs(lattice - alpha0))
    offsets = np.arange(max(-half_width, -centre), min(half_width, len(lattice) - 1 - centre) + 1)
    slices = [scd_slice(Sx, f, alpha, lattice[centre + k], n_f, (lattice, step)) for k in offsets]
    if db:
        top = 10 * np.log10(max(np.max(values) for values, _, _ in slices))
        floor = top - dynamic_range
        slices = [(np.maximum(10 * np.log10(values), floor), f_slice, a) for values, f_slice, a in slices]

    if ax is None:
        ax = plt.figure(figsize=(7, 4.6)).add_subplot(projection="3d")
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(offsets)))
    for k, (values, f_slice, _), color in zip(offsets, slices, colors):
        ax.plot(f_slice, np.full_like(f_slice, k), values, color="black" if k == 0 else color,
                linewidth=1.1 if k == 0 else 0.6, zorder=2 if k == 0 else 1)
    ax.set_xlabel("f")
    ax.set_ylabel(r"$(\alpha - \alpha_0)\,/\,\Delta\alpha$")
    ax.set_zlabel("|S| (dB)" if db else "|S|")
    ax.view_init(elev=25, azim=-62)
    ax.set_title(rf"Slices around $\alpha_0$ = {lattice[centre]:.4g}, $\Delta\alpha$ = 1/{1 / step:.0f}", fontsize=9)
    return ax


_EXPLORER = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-__PLOTLYJS__.min.js"></script>
<style>
body { font-family: system-ui, sans-serif; margin: 16px; color: #222; background: #fff; }
#controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
#slider { flex: 1; min-width: 280px; }
#readout { font-variant-numeric: tabular-nums; min-width: 20em; }
.hint { color: #666; font-size: 13px; }
</style></head><body>
<h3 style="margin:0 0 4px">__TITLE__</h3>
<div class="hint">Arrow keys step 1/N (Shift: 10/N). Click the profile to jump there.</div>
<div id="profile" style="height:220px"></div>
<div id="controls">
  <button id="prevSpike">&#9664; spike</button><button id="prev">&#9664;</button>
  <input id="slider" type="range" min="0" step="1">
  <button id="next">&#9654;</button><button id="nextSpike">spike &#9654;</button>
  <label>&alpha; <input id="goto" type="number" step="any" style="width:7em"></label>
  <label><input id="db" type="checkbox"> dB</label>
  <label><input id="fixed" type="checkbox" checked> fixed y-axis</label>
  <span id="readout"></span>
</div>
<div id="slice" style="height:380px"></div>
<script>
const D = __DATA__;
const bytes = Uint8Array.from(atob(D.grid), c => c.charCodeAt(0));
const q = new Int16Array(bytes.buffer);  // 100 x dB, -32768 where the estimator has no estimate
const nA = D.alpha.length, nF = D.f.length;
let qMax = -32767;
for (const v of q) if (v > qMax) qMax = v;
const row = (k, db) => Array.from(q.subarray(k * nF, (k + 1) * nF),
  v => v === -32768 ? null : (db ? v / 100 : Math.pow(10, v / 1000)));
const slider = document.getElementById("slider"), dbBox = document.getElementById("db"),
      fixedBox = document.getElementById("fixed");
slider.max = nA - 1;
const profileTrace = {x: D.alpha, y: D.profile, mode: "lines", line: {color: "#222", width: 1},
  hovertemplate: "&alpha; %{x:.5f}<br>%{y:.1f} dB<extra></extra>"};
Plotly.newPlot("profile", [profileTrace], {margin: {l: 55, r: 15, t: 25, b: 35}, title: {text: "Cycle-frequency profile max_f |S| (dB)", font: {size: 13}},
  xaxis: {title: {text: "&alpha;"}}, yaxis: {title: {text: "dB"}}}, {displaylogo: false, responsive: true});
Plotly.newPlot("slice", [{x: D.f, y: row(0, false), mode: "lines", connectgaps: true, line: {color: "#1f6fb4", width: 1.5},
  hovertemplate: "f %{x:.4f}<br>%{y:.4g}<extra></extra>"}],
  {margin: {l: 55, r: 15, t: 10, b: 40}, xaxis: {title: {text: "f"}, range: [D.f[0], D.f[nF - 1]]}, yaxis: {title: {text: "|S|"}}},
  {displaylogo: false, responsive: true});
let k = 0;
function show(index) {
  k = Math.max(0, Math.min(nA - 1, index));
  slider.value = k;
  const db = dbBox.checked, a = D.alpha[k];
  const top = db ? qMax / 100 : Math.pow(10, qMax / 1000);
  const yaxis = {title: {text: db ? "|S| (dB)" : "|S|"}, autorange: !fixedBox.checked,
                 range: fixedBox.checked ? (db ? [top - D.range, top + 1] : [0, top * 1.05]) : undefined};
  Plotly.update("slice", {y: [row(k, db)]}, {yaxis: yaxis});
  Plotly.relayout("profile", {shapes: [{type: "line", x0: a, x1: a, yref: "paper", y0: 0, y1: 1,
                                        line: {color: "#d62728", width: 1}}]});
  document.getElementById("readout").textContent =
    `α = ${a.toFixed(6)}  (${Math.round(a / D.step)} × 1/${Math.round(1 / D.step)})`;
}
const nearest = a => D.alpha.reduce((best, x, i) => Math.abs(x - a) < Math.abs(D.alpha[best] - a) ? i : best, 0);
slider.oninput = () => show(+slider.value);
document.getElementById("prev").onclick = () => show(k - 1);
document.getElementById("next").onclick = () => show(k + 1);
document.getElementById("prevSpike").onclick = () => { const s = D.spikes.filter(i => i < k); if (s.length) show(s[s.length - 1]); };
document.getElementById("nextSpike").onclick = () => { const s = D.spikes.find(i => i > k); if (s !== undefined) show(s); };
document.getElementById("goto").onchange = e => show(nearest(+e.target.value));
dbBox.onchange = fixedBox.onchange = () => show(k);
document.getElementById("profile").on("plotly_click", e => show(nearest(e.points[0].x)));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target !== slider) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    show(k + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1));
  }
});
show(D.start);
</script></body></html>
"""


def scd_slice_explorer(Sx, f, alpha, path, n_f=128, dynamic_range=40, title="SCD slice explorer"):
    """Standalone HTML page that steps through the slices one cycle frequency (1/N) at a time.

    A slider, the arrow keys or a typed alpha move along the estimator's lattice; the cycle-frequency
    profile above marks the slice shown (a click on it jumps there), and the spike buttons jump between
    its prominent spikes. The slices are max-pooled onto n_f frequency bins and stored as 16-bit dB, about
    2 x n_f bytes per cycle frequency (1 MB for N = 4096 at n_f = 128). The page loads plotly.js from its CDN.
    """
    import base64
    import json
    from plotly.offline import get_plotlyjs_version

    grid, f_centres, lattice = lattice_slices(Sx, f, alpha, n_f)
    db = 10 * np.log10(np.maximum(grid, np.finfo(float).tiny))
    quantised = np.where(np.isfinite(grid), np.clip(np.round(100 * db), -32767, 32767), -32768).astype("<i2")
    profile = np.nanmax(grid, axis=1)
    spikes = significant_cycle_frequencies(profile, lattice, n_slices=100, conjugate=True, min_prominence=6)
    data = {"alpha": np.round(lattice, 8).tolist(), "f": np.round(f_centres, 6).tolist(),
            "profile": np.round(_to_db(profile, 80)[0], 2).tolist(), "step": float(lattice[1] - lattice[0]),
            "range": dynamic_range, "grid": base64.b64encode(quantised.tobytes()).decode("ascii"),
            "spikes": sorted(int(np.argmin(np.abs(lattice - a))) for a in spikes),
            "start": int(np.argmin(np.abs(lattice)))}
    page = (_EXPLORER.replace("__TITLE__", title).replace("__PLOTLYJS__", get_plotlyjs_version())
            .replace("__DATA__", json.dumps(data)))
    with open(path, "w", encoding="utf-8") as file:
        file.write(page)
    return path


