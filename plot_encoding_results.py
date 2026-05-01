"""
Diagnostic plots for the fitted linear encoding model.

Plots produced
--------------
plots/kernel_psth/
    One PDF per (behavioral feature × event type), all mice.
    2-row × 3-column grid: rows = wavelength (470 blue / 560 red), cols = ROI.

plots/kernel_psth_groups/
    Same layout, but Resilient vs Susceptible group lines only.

plots/cv_alpha/
    One PDF per behavioral feature: per-mouse CV curves (alpha vs MSE),
    with the chosen alpha marked by a star.  Layout matches the 2×3 grid.

plots/photometry_psth/
    One PDF per behavioral feature: PSTH of original (gray) and reconstructed
    (signal colour) photometry conditioned on event onsets.
    Window = −20 s … +60 s.  Per-mouse means shown faintly; group mean ± SD bold.

plots/photometry_traces/
    One PDF per mouse: original (black) vs reconstructed (green) time series.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


# ── Layout constants ──────────────────────────────────────────────────────────
ROIS        = ['mPFC', 'NAc', 'BLA']
WAVELENGTHS = ['470', '560']

WAVELENGTH_COLOR = {'470': '#377eb8', '560': '#e41a1c'}
WAVELENGTH_LABEL = {'470': '5-HT (470 nm)', '560': 'DA (560 nm)'}

# ── Group definitions ─────────────────────────────────────────────────────────
RESILIENT_MICE   = {'A_3', 'A_11', 'B_1', 'B_2', 'B_4', 'B_7', 'B_8', 'B_11', 'B_10'}
SUSCEPTIBLE_MICE = {'A_5', 'A_7', 'A_1', 'A_12'}
RESILIENT_COLOR   = '#9370db'
SUSCEPTIBLE_COLOR = '#46886d'

GROUP_DEFS = {
    'Resilient':   (RESILIENT_MICE,   RESILIENT_COLOR),
    'Susceptible': (SUSCEPTIBLE_MICE, SUSCEPTIBLE_COLOR),
}

# ── Reconstruction colours ────────────────────────────────────────────────────
RECON_ORIGINAL_COLOR  = '#000000'
RECON_PREDICTED_COLOR = '#14AF00'

# ── PSTH window ───────────────────────────────────────────────────────────────
PSTH_PRE_SEC  = 20.0
PSTH_POST_SEC = 60.0


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


def _parse_signal_name(signal_name: str) -> tuple:
    """Return (roi, wavelength_key) from '{ROI}_{wavelength}_dFF'."""
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
    return np.linspace(-pre_sec, kern_sec,
                       int(pre_sec * fps) + int(kern_sec * fps))


def _collect_kernels(kernels_df, feature, etype, signal_name,
                     mouse_ids=None, min_events=1):
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
        if len(k) != expected_len:
            if abs(len(k) - expected_len) <= 2:
                k = np.interp(np.linspace(0, 1, expected_len),
                              np.linspace(0, 1, len(k)), k)
            else:
                continue
        kernels.append(k)
    return kernels, t


def _draw_kernel(ax, t, kernels, colour, label=None, show_individuals=True):
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


def _format_kernel_ax(ax, t, row_idx, col_idx, roi, wl):
    if t is not None and len(t) > 1:
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='black', linestyle='-',  linewidth=0.4, alpha=0.3)
        ax.set_xlim(t[0], t[-1])
    ax.spines['left'].set_position(('outward', 4))
    ax.spines['bottom'].set_position(('outward', 4))
    if row_idx == 0:
        ax.set_title(roi, fontsize=11, fontweight='bold')
    if col_idx == 0:
        ax.set_ylabel(f'{WAVELENGTH_LABEL.get(wl, wl)}\nKernel weight (a.u.)', fontsize=9)
    if row_idx == 1:
        ax.set_xlabel('Time from onset (s)', fontsize=9)
    ax.legend(fontsize=8)


def _make_2x3_fig(suptitle: str):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7),
                             sharex=True, sharey='row')
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1a — Kernel PSTHs, all mice
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kernel_psths(kernels_df, output_dir, min_events=1):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    if kernels_df.empty:
        return

    for feature in sorted(kernels_df['behavioral_feature'].unique()):
        for etype in sorted(kernels_df['event_type'].unique()):
            if kernels_df[(kernels_df['behavioral_feature'] == feature) &
                          (kernels_df['event_type'] == etype)].empty:
                continue

            fig, axes = _make_2x3_fig(
                f'Kernel PSTH — {feature.replace("_", " ")}  [{etype}]  (all mice)')

            for wl_idx, wl in enumerate(WAVELENGTHS):
                for roi_idx, roi in enumerate(ROIS):
                    ax = axes[wl_idx][roi_idx]
                    kernels, t = _collect_kernels(
                        kernels_df, feature, etype, f'{roi}_{wl}_dFF',
                        min_events=min_events)
                    if not kernels:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, color='gray', fontsize=9)
                    else:
                        _draw_kernel(ax, t, kernels, WAVELENGTH_COLOR[wl],
                                     show_individuals=True)
                    _format_kernel_ax(ax, t, wl_idx, roi_idx, roi, wl)

            fig.tight_layout()
            path = os.path.join(output_dir, f'{feature}__{etype}__all_mice.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1b — Kernel PSTHs, Resilient vs Susceptible
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kernel_psths_by_group(kernels_df, output_dir, min_events=1):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    if kernels_df.empty:
        return

    for feature in sorted(kernels_df['behavioral_feature'].unique()):
        for etype in sorted(kernels_df['event_type'].unique()):
            if kernels_df[(kernels_df['behavioral_feature'] == feature) &
                          (kernels_df['event_type'] == etype)].empty:
                continue

            fig, axes = _make_2x3_fig(
                f'Kernel PSTH — {feature.replace("_", " ")}  [{etype}]'
                '  (Resilient vs Susceptible)')

            for wl_idx, wl in enumerate(WAVELENGTHS):
                for roi_idx, roi in enumerate(ROIS):
                    ax      = axes[wl_idx][roi_idx]
                    ref_t   = None
                    any_data = False
                    for gname, (mids, colour) in GROUP_DEFS.items():
                        kernels, t = _collect_kernels(
                            kernels_df, feature, etype, f'{roi}_{wl}_dFF',
                            mouse_ids=mids, min_events=min_events)
                        if kernels and t is not None:
                            _draw_kernel(ax, t, kernels, colour,
                                         label=gname, show_individuals=False)
                            any_data = True
                            ref_t    = t
                    if not any_data:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, color='gray', fontsize=9)
                    _format_kernel_ax(ax, ref_t, wl_idx, roi_idx, roi, wl)

            fig.tight_layout()
            path = os.path.join(output_dir, f'{feature}__{etype}__groups.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — CV alpha vs MSE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cv_alpha_vs_mse(kernels_df, output_dir, min_events=1):
    """
    Per-behavioral-feature PDF showing each mouse's CV alpha-vs-MSE curve,
    with the chosen alpha marked by a star.  2×3 grid: rows = signal type,
    cols = ROI.
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    if kernels_df.empty:
        return

    # Only onset rows (offset would be duplicate CV info)
    df = kernels_df[kernels_df['event_type'] == 'onset'].copy()
    if 'cv_alphas' not in df.columns:
        print('  cv_alphas column missing — skipping CV alpha plots.')
        return

    signal_names = sorted(df['signal_name'].unique())

    def _signal_info(sn):
        roi, wl = _parse_signal_name(sn)
        label       = f'{roi} {WAVELENGTH_LABEL.get(wl, wl)}'
        signal_type = wl   # '470' or '560'
        return label, signal_type, roi

    for feature in sorted(df['behavioral_feature'].unique()):
        fdata = df[(df['behavioral_feature'] == feature) &
                   (df['n_events'] >= min_events)]
        if fdata.empty:
            continue

        mice    = sorted(fdata['session_id'].unique())
        n_mice  = len(mice)
        if _HAS_SEABORN:
            palette = sns.color_palette('husl', n_mice)
        else:
            cmap    = plt.cm.get_cmap('tab20', n_mice)
            palette = [cmap(i) for i in range(n_mice)]

        mouse_colour = {m: palette[i] for i, m in enumerate(mice)}

        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        fig.suptitle(f'CV Alpha vs MSE — {feature.replace("_", " ")}',
                     fontsize=13, fontweight='bold', y=1.02)

        for wl_idx, wl in enumerate(WAVELENGTHS):
            for roi_idx, roi in enumerate(ROIS):
                ax          = axes[wl_idx][roi_idx]
                signal_name = f'{roi}_{wl}_dFF'
                sig_data    = fdata[fdata['signal_name'] == signal_name]

                if sig_data.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, color='gray', fontsize=9)
                    ax.set_visible(True)
                    continue

                label = f'{roi} {WAVELENGTH_LABEL.get(wl, wl)}'

                for mouse in mice:
                    mdata = sig_data[sig_data['session_id'] == mouse]
                    colour = mouse_colour[mouse]
                    for _, row in mdata.iterrows():
                        alphas = np.asarray(row['cv_alphas'],   dtype=float)
                        mses   = np.asarray(row['cv_mse_scores'], dtype=float)
                        chosen = float(row['alpha'])

                        finite = np.isfinite(mses)
                        if not np.any(finite):
                            continue

                        sort_idx = np.argsort(alphas)
                        a_s = alphas[sort_idx]
                        m_s = mses[sort_idx]

                        # Line + scatter
                        ax.plot(a_s, m_s, color=colour, alpha=0.7,
                                linewidth=1.2,
                                label=f'Mouse {mouse}' if (wl_idx == 0 and roi_idx == 0) else None)
                        ax.scatter(a_s, m_s, color=colour, alpha=0.5, s=18, zorder=3)

                        # Star on chosen alpha
                        chosen_idx = np.searchsorted(a_s, chosen)
                        chosen_idx = min(chosen_idx, len(a_s) - 1)
                        ax.scatter(chosen, m_s[chosen_idx], color=colour,
                                   marker='*', s=120, edgecolors='black',
                                   linewidths=0.5, zorder=4)

                ax.set_xscale('log')
                ax.set_xlabel('Alpha', fontsize=9)
                ax.set_ylabel('CV MSE', fontsize=9)
                ax.set_title(label, fontsize=10, fontweight='bold')
                ax.spines['left'].set_position(('outward', 4))
                ax.spines['bottom'].set_position(('outward', 4))

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center',
                       bbox_to_anchor=(0.5, -0.02),
                       ncol=min(6, n_mice), fontsize=8, frameon=False)

        fig.tight_layout()
        path = os.path.join(output_dir, f'{feature}__cv_alpha.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — PSTH: original vs reconstructed, conditioned on event onsets
# ═══════════════════════════════════════════════════════════════════════════════

def _find_onsets(binary_signal: np.ndarray) -> np.ndarray:
    """Return sample indices of 0→1 transitions (handles NaN by treating as 0)."""
    sig = np.asarray(binary_signal, dtype=float).copy()
    sig[np.isnan(sig)] = 0.0
    diff    = np.diff(sig.astype(int))
    onsets  = np.where(diff == 1)[0] + 1
    if len(sig) > 0 and sig[0] == 1:
        onsets = np.concatenate([[0], onsets])
    return onsets.astype(int)


def _extract_snippets(signal: np.ndarray,
                      onsets: np.ndarray,
                      pre_frames: int,
                      post_frames: int) -> np.ndarray | None:
    """
    Return array of shape (n_valid_events, pre_frames+post_frames) by cutting
    snippets [onset-pre_frames : onset+post_frames] from signal.
    Snippets containing any NaN or falling outside signal bounds are discarded.
    """
    total = pre_frames + post_frames
    snips = []
    for idx in onsets:
        start = idx - pre_frames
        end   = idx + post_frames
        if start < 0 or end > len(signal):
            continue
        snip = signal[start:end]
        if np.any(np.isnan(snip)):
            continue
        snips.append(snip)
    return np.vstack(snips) if snips else None


def _collect_mouse_psths(reconstruction_data: dict,
                         feature: str,
                         signal_name: str) -> dict:
    """
    For each mouse that has (mouse_id, signal_name) in reconstruction_data,
    compute per-mouse mean PSTHs for both original and reconstructed signals.

    Returns dict:
        { mouse_id: {'original': 1-D array, 'reconstructed': 1-D array,
                     'fps': float, 'n_events': int} }
    """
    result = {}
    for (mouse_id, sig), info in reconstruction_data.items():
        if sig != signal_name:
            continue
        beh_sigs = info.get('behavioral_signals', {})
        if feature not in beh_sigs:
            continue

        fps        = float(info['effective_fps'])
        pre_frames = int(PSTH_PRE_SEC  * fps)
        post_frames= int(PSTH_POST_SEC * fps)

        onsets = _find_onsets(beh_sigs[feature])
        if len(onsets) == 0:
            continue

        original    = np.asarray(info['original_signal'], dtype=float)
        predicted   = (info['design_matrix'] @ info['coefficients']
                       + info['intercept'])

        orig_snips  = _extract_snippets(original,  onsets, pre_frames, post_frames)
        pred_snips  = _extract_snippets(predicted, onsets, pre_frames, post_frames)

        if orig_snips is None and pred_snips is None:
            continue

        result[mouse_id] = {
            'original':      np.nanmean(orig_snips,  axis=0) if orig_snips  is not None else None,
            'reconstructed': np.nanmean(pred_snips,  axis=0) if pred_snips  is not None else None,
            'fps':           fps,
            'n_events':      len(onsets),
        }
    return result


def plot_photometry_psths(reconstruction_data: dict,
                          kernels_df: pd.DataFrame,
                          output_dir: str) -> None:
    """
    For each (behavioral feature × event type), produce a 2×3 PDF showing the
    PSTH of original (gray) and reconstructed (signal colour) photometry,
    averaged per mouse.  Individual per-mouse traces are faint; mean ± SD bold.
    Window: −20 s to +60 s relative to event onset.
    """
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    if not reconstruction_data:
        print('  reconstruction_data empty — skipping PSTH plots.')
        return

    features = sorted(kernels_df['behavioral_feature'].unique()) if not kernels_df.empty else []
    if not features:
        # Fall back: collect features from reconstruction_data
        for _, info in reconstruction_data.items():
            features = list(info.get('behavioral_features', []))
            if features:
                break

    for feature in features:
        fig, axes = _make_2x3_fig(
            f'Photometry PSTH — {feature.replace("_", " ")}  '
            f'(−{PSTH_PRE_SEC:.0f} s … +{PSTH_POST_SEC:.0f} s from onset)')

        any_panel_data = False

        for wl_idx, wl in enumerate(WAVELENGTHS):
            for roi_idx, roi in enumerate(ROIS):
                ax          = axes[wl_idx][roi_idx]
                signal_name = f'{roi}_{wl}_dFF'
                colour      = WAVELENGTH_COLOR[wl]

                mouse_psths = _collect_mouse_psths(
                    reconstruction_data, feature, signal_name)

                if not mouse_psths:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, color='gray', fontsize=9)
                    _format_psth_ax(ax, wl_idx, roi_idx, roi, wl)
                    continue

                # Build per-mouse arrays; use the fps from the first mouse
                fps = next(iter(mouse_psths.values()))['fps']
                pre_frames  = int(PSTH_PRE_SEC  * fps)
                post_frames = int(PSTH_POST_SEC * fps)
                t_psth = np.linspace(-PSTH_PRE_SEC, PSTH_POST_SEC,
                                     pre_frames + post_frames)

                orig_list  = [v['original']      for v in mouse_psths.values()
                              if v['original']      is not None]
                recon_list = [v['reconstructed'] for v in mouse_psths.values()
                              if v['reconstructed'] is not None]

                def _plot_group(traces, col, alpha_ind, alpha_band, lbl):
                    if not traces:
                        return
                    arr  = np.vstack(traces)
                    mean = np.nanmean(arr, axis=0)
                    sd   = np.nanstd(arr, axis=0, ddof=min(1, arr.shape[0]-1))
                    for tr in traces:
                        ax.plot(t_psth, tr, color=col, alpha=alpha_ind,
                                linewidth=0.6, zorder=1)
                    ax.plot(t_psth, mean, color=col, linewidth=2.0,
                            label=lbl, zorder=3)
                    ax.fill_between(t_psth, mean - sd, mean + sd,
                                    color=col, alpha=alpha_band, zorder=2)

                _plot_group(orig_list,  'gray', 0.15, 0.15, 'Original')
                _plot_group(recon_list, colour, 0.15, 0.25, 'Reconstructed')

                any_panel_data = True
                _format_psth_ax(ax, wl_idx, roi_idx, roi, wl)

        if not any_panel_data:
            plt.close(fig)
            continue

        fig.tight_layout()
        path = os.path.join(output_dir, f'{feature}__psth.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


def _format_psth_ax(ax, row_idx, col_idx, roi, wl):
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='black', linestyle='-',  linewidth=0.4, alpha=0.3)
    ax.set_xlim(-PSTH_PRE_SEC, PSTH_POST_SEC)
    ax.spines['left'].set_position(('outward', 4))
    ax.spines['bottom'].set_position(('outward', 4))
    if row_idx == 0:
        ax.set_title(roi, fontsize=11, fontweight='bold')
    if col_idx == 0:
        ax.set_ylabel(f'{WAVELENGTH_LABEL.get(wl, wl)}\nΔF/F', fontsize=9)
    if row_idx == 1:
        ax.set_xlabel('Time from onset (s)', fontsize=9)
    ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Original vs reconstructed photometry traces (per mouse)
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_signal(info: dict) -> tuple:
    """
    Returns (time_axis, original_signal, predicted_signal, mse_score).
    design_matrix is already the spline design matrix, so
    predicted = X_spline @ spline_coef + intercept.
    """
    predicted = info['design_matrix'] @ info['coefficients'] + info['intercept']
    fps       = float(info['effective_fps'])
    time_axis = np.arange(len(info['original_signal'])) / fps
    return time_axis, info['original_signal'], predicted, info['mse_score']


def plot_signal_reconstructions(reconstruction_data: dict,
                                output_dir: str,
                                time_range: tuple = None) -> None:
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)
    if not reconstruction_data:
        return

    by_mouse: dict = {}
    for mouse_id, signal_name in reconstruction_data:
        by_mouse.setdefault(mouse_id, []).append(signal_name)

    def _sort_key(sig):
        roi, wl = _parse_signal_name(sig)
        return (ROIS.index(roi) if roi in ROIS else 999,
                WAVELENGTHS.index(wl) if wl in WAVELENGTHS else 999)

    for mouse_id, sigs in sorted(by_mouse.items()):
        sigs  = sorted(sigs, key=_sort_key)
        n_sig = len(sigs)

        fig, axes = plt.subplots(n_sig, 1, figsize=(16, 2.5 * n_sig), squeeze=False)
        fig.suptitle(f'Original vs Reconstructed — Mouse {mouse_id}',
                     fontsize=13, fontweight='bold', y=1.01)

        for i, signal_name in enumerate(sigs):
            ax   = axes[i, 0]
            info = reconstruction_data[(mouse_id, signal_name)]
            t, original, predicted, mse = reconstruct_signal(info)

            if time_range is not None:
                mask      = (t >= time_range[0]) & (t <= time_range[1])
                t         = t[mask]
                original  = original[mask]
                predicted = predicted[mask]

            ax.plot(t, original,  color=RECON_ORIGINAL_COLOR,
                    alpha=0.7,  linewidth=0.8, label='Original',      zorder=1)
            ax.plot(t, predicted, color=RECON_PREDICTED_COLOR,
                    alpha=0.85, linewidth=0.8, label='Reconstructed', zorder=2)

            roi, wl   = _parse_signal_name(signal_name)
            sig_label = WAVELENGTH_LABEL.get(wl, signal_name.replace('_dFF', ''))
            ax.set_ylabel(f'{roi}\n{sig_label}\nΔF/F', fontsize=9)

            n_sess  = info.get('n_sessions', 1)
            title   = f'{roi} — {sig_label}   MSE = {mse:.4f}'
            if n_sess > 1:
                title += f'  [{n_sess} sessions concatenated]'
            ax.set_title(title, fontsize=10, loc='left')

            if i == n_sig - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
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
    p = argparse.ArgumentParser()
    p.add_argument('--kernels_pkl',    required=True)
    p.add_argument('--recon_data_pkl', required=True,
                   help='Path to reconstruction_data.pkl')
    p.add_argument('--output_dir',     default='plots')
    p.add_argument('--min_events',     type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    print('Loading results …')
    with open(args.kernels_pkl, 'rb') as fh:
        kernels_df: pd.DataFrame = pickle.load(fh)
    with open(args.recon_data_pkl, 'rb') as fh:
        reconstruction_data: dict = pickle.load(fh)

    psth_dir    = os.path.join(args.output_dir, 'kernel_psth')
    groups_dir  = os.path.join(args.output_dir, 'kernel_psth_groups')
    cv_dir      = os.path.join(args.output_dir, 'cv_alpha')
    phot_psth_dir = os.path.join(args.output_dir, 'photometry_psth')
    traces_dir  = os.path.join(args.output_dir, 'photometry_traces')

    print(f'\n── All-mice kernel PSTHs → {psth_dir}')
    plot_kernel_psths(kernels_df, psth_dir, min_events=args.min_events)

    print(f'\n── Group kernel PSTHs → {groups_dir}')
    plot_kernel_psths_by_group(kernels_df, groups_dir, min_events=args.min_events)

    print(f'\n── CV alpha vs MSE plots → {cv_dir}')
    plot_cv_alpha_vs_mse(kernels_df, cv_dir, min_events=args.min_events)

    print(f'\n── Photometry PSTHs (original vs reconstructed) → {phot_psth_dir}')
    plot_photometry_psths(reconstruction_data, kernels_df, phot_psth_dir)

    print(f'\n── Photometry trace plots → {traces_dir}')
    plot_signal_reconstructions(reconstruction_data, traces_dir)

    print('\nAll plots saved.')


if __name__ == '__main__':
    main()
