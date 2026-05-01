"""
Diagnostic plots for the fitted linear encoding model.

Plots produced
--------------
plots/kernel_psth/
    One PDF per (behavioral feature × event type), all mice.
    Fixed 2-row × 3-column grid:
      rows → wavelength  (row 0 = 470 nm / 5-HT blue, row 1 = 560 nm / DA red)
      cols → ROI         (col 0 = mPFC, col 1 = NAc, col 2 = BLA)
    Individual mouse traces shown faintly; bold mean ± shaded SEM.

plots/kernel_psth_groups/
    Same 2×3 layout, but two lines per panel:
      Resilient   — purple (#9370db)
      Susceptible — teal   (#46886d)
    No individual traces (to keep it readable).

plots/photometry_traces/
    One PDF per mouse: original (black) vs reconstructed (green) traces,
    one panel per signal sorted by ROI then wavelength.

Usage
-----
    python plot_encoding_results.py \
        --kernels_pkl    kernels_df.pkl \
        --recon_data_pkl reconstruction_data.pkl \
        --output_dir     plots
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")          # non-interactive — safe on cluster nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem


# ── Grid layout ───────────────────────────────────────────────────────────────
ROIS       = ['mPFC', 'NAc', 'BLA']   # subplot columns (left → right)
WAVELENGTHS = ['470', '560']           # subplot rows (top → bottom)

WAVELENGTH_COLOR = {
    '470': '#377eb8',   # blue  — 5-HT
    '560': '#e41a1c',   # red   — DA
}
WAVELENGTH_LABEL = {
    '470': '5-HT (470 nm)',
    '560': 'DA (560 nm)',
}

# ── Mouse group definitions ───────────────────────────────────────────────────
RESILIENT_MICE   = {'A_3', 'A_11', 'B_1', 'B_2', 'B_4', 'B_7', 'B_8', 'B_11', 'B_10'}
SUSCEPTIBLE_MICE = {'A_5', 'A_7', 'A_1', 'A_12'}

RESILIENT_COLOR   = '#9370db'   # purple
SUSCEPTIBLE_COLOR = '#46886d'   # teal-green

GROUP_DEFS = {
    'Resilient':   (RESILIENT_MICE,   RESILIENT_COLOR),
    'Susceptible': (SUSCEPTIBLE_MICE, SUSCEPTIBLE_COLOR),
}

# ── Reconstruction plot colours ───────────────────────────────────────────────
RECON_ORIGINAL_COLOR   = '#000000'   # black
RECON_PREDICTED_COLOR  = '#14AF00'   # green


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_style() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'figure.dpi': 150,
    })


def _parse_signal_name(signal_name: str) -> tuple[str, str | None]:
    """
    Parse a signal name of the form '{ROI}_{wavelength}_dFF'.
    Returns (roi, wavelength_key) where wavelength_key is '470', '560', or None.
    Also handles name-based conventions (DA → '560', 5HT / 5-HT → '470').
    """
    base  = signal_name.replace('_dFF', '')
    parts = base.split('_')
    roi   = parts[0]
    rest  = '_'.join(parts[1:])

    if '560' in rest:
        return roi, '560'
    if '470' in rest:
        return roi, '470'
    if 'DA' in rest.upper() and '5' not in rest:
        return roi, '560'
    if '5HT' in rest.upper() or '5-HT' in rest.upper():
        return roi, '470'
    return roi, None


def _make_time_vector(ref_row: pd.Series) -> np.ndarray:
    fps      = float(ref_row['effective_fps'])
    pre_sec  = float(ref_row['pre_onset_sec'])
    kern_sec = float(ref_row['kernel_length_sec'])
    n_pre    = int(pre_sec  * fps)
    n_kern   = int(kern_sec * fps)
    return np.linspace(-pre_sec, kern_sec, n_pre + n_kern)


def _collect_kernels(
    kernels_df:  pd.DataFrame,
    feature:     str,
    etype:       str,
    signal_name: str,
    mouse_ids:   set | None = None,
    min_events:  int = 1,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """
    Return (list_of_kernel_arrays, time_vector) for the given combination.
    Optionally restrict to a set of mouse IDs.
    Returns ([], None) when nothing passes the filters.
    """
    mask = (
        (kernels_df['behavioral_feature'] == feature) &
        (kernels_df['event_type']         == etype)   &
        (kernels_df['signal_name']        == signal_name) &
        (kernels_df['n_events']           >= min_events)
    )
    subset = kernels_df[mask]
    if mouse_ids is not None:
        subset = subset[subset['session_id'].isin(mouse_ids)]
    if subset.empty:
        return [], None

    t            = _make_time_vector(subset.iloc[0])
    expected_len = len(t)
    kernels      = []

    for _, row in subset.iterrows():
        k = row['kernel']
        if k is None:
            continue
        k = np.asarray(k, dtype=np.float64)
        if k.ndim != 1:
            continue
        if len(k) != expected_len:
            # Tolerate rounding differences of ≤2 frames via linear interpolation
            if abs(len(k) - expected_len) <= 2:
                k = np.interp(
                    np.linspace(0, 1, expected_len),
                    np.linspace(0, 1, len(k)),
                    k,
                )
            else:
                continue
        kernels.append(k)

    return kernels, t


def _draw_kernel(
    ax:               plt.Axes,
    t:                np.ndarray,
    kernels:          list[np.ndarray],
    colour:           str,
    label:            str | None = None,
    show_individuals: bool = True,
) -> None:
    """Plot individual traces (optional), mean, and SEM band on ax."""
    if not kernels:
        return

    arr    = np.vstack(kernels)
    mean_k = np.nanmean(arr, axis=0)
    n      = arr.shape[0]

    if show_individuals:
        for k in kernels:
            ax.plot(t, k, color=colour, alpha=0.15, linewidth=0.7, zorder=1)

    lbl = f'{label} (n={n})' if label else f'n={n}'
    ax.plot(t, mean_k, color=colour, linewidth=2.0, label=lbl, zorder=3)

    if n > 1:
        sem_k = scipy_sem(arr, axis=0, nan_policy='omit')
        ax.fill_between(t, mean_k - sem_k, mean_k + sem_k,
                        color=colour, alpha=0.25, zorder=2)


def _format_kernel_ax(
    ax:      plt.Axes,
    t:       np.ndarray | None,
    row_idx: int,
    col_idx: int,
    roi:     str,
    wl:      str,
) -> None:
    """Apply consistent formatting to one kernel subplot."""
    if t is not None and len(t) > 1:
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='black', linestyle='-',  linewidth=0.4, alpha=0.3)
        ax.set_xlim(t[0], t[-1])

    ax.spines['left'].set_position(('outward', 4))
    ax.spines['bottom'].set_position(('outward', 4))

    # Column header (ROI) on the top row only
    if row_idx == 0:
        ax.set_title(roi, fontsize=11, fontweight='bold')

    # Y-axis label on the leftmost column only
    if col_idx == 0:
        ax.set_ylabel(f'{WAVELENGTH_LABEL.get(wl, wl)}\nKernel weight (a.u.)', fontsize=9)

    # X-axis label on the bottom row only
    if row_idx == 1:
        ax.set_xlabel('Time from onset (s)', fontsize=9)

    ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1a — Kernel PSTHs, all mice
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kernel_psths(
    kernels_df: pd.DataFrame,
    output_dir: str,
    min_events: int = 1,
) -> None:
    """
    For each (behavioral feature × event type), save a 2×3 PDF showing
    mean ± SEM kernels for all mice, with individual traces in the background.
    Rows = wavelength (470 / 560), columns = ROI (mPFC, NAc, BLA).
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if kernels_df.empty:
        print('  kernels_df is empty — skipping all-mice kernel plots.')
        return

    features = sorted(kernels_df['behavioral_feature'].unique())
    etypes   = sorted(kernels_df['event_type'].unique())

    for feature in features:
        for etype in etypes:

            combo = kernels_df[
                (kernels_df['behavioral_feature'] == feature) &
                (kernels_df['event_type']         == etype)
            ]
            if combo.empty:
                continue

            fig, axes = plt.subplots(
                2, 3,
                figsize=(13, 7),
                sharex=True,
                sharey='row',
            )
            fig.suptitle(
                f'Kernel PSTH — {feature.replace("_", " ")}  [{etype}]  (all mice)',
                fontsize=13, fontweight='bold', y=1.02,
            )

            for wl_idx, wl in enumerate(WAVELENGTHS):
                for roi_idx, roi in enumerate(ROIS):
                    ax          = axes[wl_idx][roi_idx]
                    signal_name = f'{roi}_{wl}_dFF'
                    colour      = WAVELENGTH_COLOR[wl]

                    kernels, t = _collect_kernels(
                        kernels_df, feature, etype, signal_name,
                        min_events=min_events,
                    )

                    if not kernels:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, color='gray', fontsize=9)
                    else:
                        _draw_kernel(ax, t, kernels, colour, show_individuals=True)

                    _format_kernel_ax(ax, t, wl_idx, roi_idx, roi, wl)

            fig.tight_layout()
            fname = f'{feature}__{etype}__all_mice.pdf'
            path  = os.path.join(output_dir, fname)
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1b — Kernel PSTHs, Resilient vs Susceptible
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kernel_psths_by_group(
    kernels_df: pd.DataFrame,
    output_dir: str,
    min_events: int = 1,
) -> None:
    """
    Same 2×3 layout as plot_kernel_psths, but overlays separate mean ± SEM
    lines for the Resilient and Susceptible mouse groups (no individual traces).
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if kernels_df.empty:
        print('  kernels_df is empty — skipping group kernel plots.')
        return

    features = sorted(kernels_df['behavioral_feature'].unique())
    etypes   = sorted(kernels_df['event_type'].unique())

    for feature in features:
        for etype in etypes:

            combo = kernels_df[
                (kernels_df['behavioral_feature'] == feature) &
                (kernels_df['event_type']         == etype)
            ]
            if combo.empty:
                continue

            fig, axes = plt.subplots(
                2, 3,
                figsize=(13, 7),
                sharex=True,
                sharey='row',
            )
            fig.suptitle(
                f'Kernel PSTH — {feature.replace("_", " ")}  [{etype}]'
                '  (Resilient vs Susceptible)',
                fontsize=13, fontweight='bold', y=1.02,
            )

            for wl_idx, wl in enumerate(WAVELENGTHS):
                for roi_idx, roi in enumerate(ROIS):
                    ax          = axes[wl_idx][roi_idx]
                    signal_name = f'{roi}_{wl}_dFF'
                    any_data    = False
                    ref_t       = None

                    for group_name, (mouse_ids, colour) in GROUP_DEFS.items():
                        kernels, t = _collect_kernels(
                            kernels_df, feature, etype, signal_name,
                            mouse_ids=mouse_ids,
                            min_events=min_events,
                        )
                        if kernels and t is not None:
                            _draw_kernel(ax, t, kernels, colour,
                                         label=group_name, show_individuals=False)
                            any_data = True
                            ref_t    = t   # keep a valid time vector for formatting

                    if not any_data:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, color='gray', fontsize=9)

                    _format_kernel_ax(ax, ref_t, wl_idx, roi_idx, roi, wl)

            fig.tight_layout()
            fname = f'{feature}__{etype}__groups.pdf'
            path  = os.path.join(output_dir, fname)
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Original vs reconstructed photometry traces
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_signal(reconstruction_info: dict) -> tuple:
    """
    Reconstruct the predicted signal from stored encoding model weights.

    Parameters
    ----------
    reconstruction_info : dict
        One entry from the reconstruction_data dict returned by
        fit_linear_encoding_models, keyed by (mouse_id, signal_name).

    Returns
    -------
    time_axis       : np.ndarray  — seconds
    original_signal : np.ndarray  — may contain NaN where data was masked
    predicted_signal: np.ndarray  — X @ coef + intercept (no NaN gaps)
    mse_score       : float
    """
    X            = reconstruction_info['design_matrix']
    coefficients = reconstruction_info['coefficients']
    intercept    = reconstruction_info['intercept']
    original     = reconstruction_info['original_signal']
    fps          = reconstruction_info['effective_fps']

    predicted = X @ coefficients + intercept
    time_axis = np.arange(len(original)) / fps
    mse_score = reconstruction_info['mse_score']

    return time_axis, original, predicted, mse_score


def plot_signal_reconstructions(
    reconstruction_data: dict,
    output_dir:          str,
    time_range:          tuple | None = None,
    figsize_per_panel:   tuple = (16, 2.5),
    alpha_original:      float = 0.7,
    alpha_predicted:     float = 0.85,
) -> None:
    """
    For each mouse in reconstruction_data, save one PDF with one panel per
    signal, ordered by ROI (mPFC → NAc → BLA) then wavelength (470 → 560).

    Parameters
    ----------
    time_range : (start_sec, end_sec) or None
        If given, crop all traces to this window.
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if not reconstruction_data:
        print('  reconstruction_data is empty — skipping trace plots.')
        return

    # ── group keys by mouse ──────────────────────────────────────────────────
    by_mouse: dict[str, list[str]] = {}
    for mouse_id, signal_name in reconstruction_data.keys():
        by_mouse.setdefault(mouse_id, []).append(signal_name)

    def _sort_signal(sig: str) -> tuple:
        roi, wl = _parse_signal_name(sig)
        roi_idx = ROIS.index(roi)       if roi in ROIS       else 999
        wl_idx  = WAVELENGTHS.index(wl) if wl in WAVELENGTHS else 999
        return (roi_idx, wl_idx)

    for mouse_id, signal_names in sorted(by_mouse.items()):
        signal_names = sorted(signal_names, key=_sort_signal)
        n_sig = len(signal_names)

        fig, axes = plt.subplots(
            n_sig, 1,
            figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_sig),
            squeeze=False,
        )
        fig.suptitle(
            f'Encoding Model: Original vs Reconstructed — Mouse {mouse_id}',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for i, signal_name in enumerate(signal_names):
            ax   = axes[i, 0]
            info = reconstruction_data[(mouse_id, signal_name)]

            time_axis, original, predicted, mse = reconstruct_signal(info)

            # Optional crop
            if time_range is not None:
                mask      = (time_axis >= time_range[0]) & (time_axis <= time_range[1])
                time_axis = time_axis[mask]
                original  = original[mask]
                predicted = predicted[mask]

            ax.plot(time_axis, original,  color=RECON_ORIGINAL_COLOR,
                    alpha=alpha_original,  linewidth=0.8,
                    label='Original',  zorder=1)
            ax.plot(time_axis, predicted, color=RECON_PREDICTED_COLOR,
                    alpha=alpha_predicted, linewidth=0.8,
                    label='Reconstructed', zorder=2)

            # ── axis labels ──────────────────────────────────────────────────
            roi, wl        = _parse_signal_name(signal_name)
            sig_label      = WAVELENGTH_LABEL.get(wl, signal_name.replace('_dFF', ''))
            ax.set_ylabel(f'{roi}\n{sig_label}\nΔF/F', fontsize=9)

            if i == n_sig - 1:
                ax.set_xlabel('Time (s)', fontsize=10)

            # ── panel title ──────────────────────────────────────────────────
            n_sessions = info.get('n_sessions', 1)
            title      = f'{roi} — {sig_label}   MSE = {mse:.4f}'
            if n_sessions > 1:
                title += f'  [{n_sessions} sessions concatenated]'
            ax.set_title(title, fontsize=10, loc='left')

            if i == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)

        fig.tight_layout()
        path = os.path.join(output_dir, f'mouse_{mouse_id}.pdf')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate diagnostic plots for the fitted linear encoding model.'
    )
    p.add_argument('--kernels_pkl',    required=True,
                   help='Path to kernels_df.pkl')
    p.add_argument('--recon_data_pkl', required=True,
                   help='Path to reconstruction_data.pkl '
                        '(dict keyed by (mouse_id, signal_name))')
    p.add_argument('--output_dir',     default='plots',
                   help='Root directory for output PDFs')
    p.add_argument('--min_events',     type=int, default=1,
                   help='Minimum events for a mouse/group to appear in kernel plots')
    return p.parse_args()


def main():
    args = parse_args()

    print('Loading results …')
    with open(args.kernels_pkl, 'rb') as fh:
        kernels_df: pd.DataFrame = pickle.load(fh)
    with open(args.recon_data_pkl, 'rb') as fh:
        reconstruction_data: dict = pickle.load(fh)

    psth_dir   = os.path.join(args.output_dir, 'kernel_psth')
    groups_dir = os.path.join(args.output_dir, 'kernel_psth_groups')
    traces_dir = os.path.join(args.output_dir, 'photometry_traces')

    print(f'\n── All-mice kernel PSTHs → {psth_dir}')
    plot_kernel_psths(kernels_df, psth_dir, min_events=args.min_events)

    print(f'\n── Group kernel PSTHs (Resilient vs Susceptible) → {groups_dir}')
    plot_kernel_psths_by_group(kernels_df, groups_dir, min_events=args.min_events)

    print(f'\n── Photometry trace plots → {traces_dir}')
    plot_signal_reconstructions(reconstruction_data, traces_dir)

    print('\nAll plots saved.')


if __name__ == '__main__':
    main()
