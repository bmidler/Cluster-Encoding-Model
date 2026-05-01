"""
Worker script: fit the linear encoding model for a single mouse.
Called by fit_encoding_array.sh via SLURM_ARRAY_TASK_ID.

Usage:
    python fit_encoding_worker.py \
        --mice_list mice_list.txt \
        --task_id 0 \
        --photometry_pkl segmentation_data.pkl \
        --annotations_pkl dff_data.pkl \
        --output_dir results
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# ── model import ────────────────────────────────────────────────────────────
# EncodingModelTemplate.py must be on the path (same directory by default).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EncodingModelTemplate import fit_linear_encoding_models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mice_list",       required=True,
                   help="Path to mice_list.txt (one mouse_id per line)")
    p.add_argument("--task_id",         type=int, required=True,
                   help="Zero-based index into mice_list (== SLURM_ARRAY_TASK_ID)")
    p.add_argument("--photometry_pkl",  required=True,
                   help="Path to dff_data.pkl (photometry DataFrame)")
    p.add_argument("--annotations_pkl", required=True,
                   help="Path to segmentation_data.pkl (annotations DataFrame)")
    p.add_argument("--output_dir",      default="results",
                   help="Directory in which to write per-mouse result pickles")
    # ── model hyper-parameters ───────────────────────────────────────────────
    p.add_argument("--kernel_length_sec", type=float, default=2.0)
    p.add_argument("--pre_onset_sec",     type=float, default=2.0)
    p.add_argument("--target_fps",        type=float, default=1.0)
    p.add_argument("--regularization",    default="ridge",
                   choices=["ridge", "lasso", "elastic_net"])
    p.add_argument("--alpha_low",         type=float, default=1.0,
                   help="log10 lower bound for alpha grid (np.logspace)")
    p.add_argument("--alpha_high",        type=float, default=4.0,
                   help="log10 upper bound for alpha grid")
    p.add_argument("--alpha_n",           type=int,   default=10,
                   help="Number of alpha values in the grid")
    p.add_argument("--l1_ratio",          type=float, default=0.5)
    p.add_argument("--offset",            action="store_true", default=False)
    p.add_argument("--session_type",      default="Defeat")
    p.add_argument("--days_to_include",   nargs="*", type=int, default=None,
                   help="Day numbers to include, e.g. --days_to_include 1 2 3")
    return p.parse_args()


def extract_mouse_id(session_id: str) -> str:
    parts = session_id.split("-")
    return parts[2] if len(parts) >= 3 else parts[0]


def filter_to_mouse(df: pd.DataFrame, mouse_id: str) -> pd.DataFrame:
    """Return rows whose index belongs to mouse_id."""
    mask = [extract_mouse_id(sid) == mouse_id for sid in df.index]
    return df[mask]


def main():
    args = parse_args()

    # ── resolve mouse_id from list ───────────────────────────────────────────
    with open(args.mice_list) as fh:
        mice = [line.strip() for line in fh if line.strip()]

    if args.task_id >= len(mice):
        print(f"task_id {args.task_id} >= number of mice ({len(mice)}). Nothing to do.")
        sys.exit(0)

    mouse_id = mice[args.task_id]
    print(f"[task {args.task_id}] Processing mouse: {mouse_id}")

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading photometry data …")
    with open(args.photometry_pkl, "rb") as fh:
        dff_data_df = pickle.load(fh)

    print("Loading annotations data …")
    with open(args.annotations_pkl, "rb") as fh:
        segmentation_binary_df = pickle.load(fh)

    # ── filter to this mouse only ────────────────────────────────────────────
    dff_mouse       = filter_to_mouse(dff_data_df,          mouse_id)
    seg_mouse       = filter_to_mouse(segmentation_binary_df, mouse_id)

    if dff_mouse.empty or seg_mouse.empty:
        print(f"No sessions found for mouse {mouse_id}. Exiting.")
        sys.exit(0)

    print(f"  Sessions: {list(dff_mouse.index)}")

    # ── build alpha grid ─────────────────────────────────────────────────────
    alpha = np.logspace(args.alpha_low, args.alpha_high, args.alpha_n)

    # ── fit ──────────────────────────────────────────────────────────────────
    kernels_df, reconstruction_dict, reconstructed_dff_df = fit_linear_encoding_models(
        photometry_df    = dff_mouse,
        annotations_df   = seg_mouse,
        kernel_length_sec= args.kernel_length_sec,
        pre_onset_sec    = args.pre_onset_sec,
        target_fps       = args.target_fps,
        regularization   = args.regularization,
        alpha            = alpha,
        l1_ratio         = args.l1_ratio,
        offset           = args.offset,
        verbose          = True,
        days_to_include  = args.days_to_include,
        session_type     = args.session_type,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"mouse_{mouse_id}.pkl")

    with open(out_path, "wb") as fh:
        pickle.dump({
            "mouse_id":           mouse_id,
            "kernels_df":         kernels_df,
            "reconstruction_dict":reconstruction_dict,
            "reconstructed_dff":  reconstructed_dff_df,
        }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
