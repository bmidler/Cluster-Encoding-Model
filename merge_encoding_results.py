"""
Merge per-mouse encoding model results into three combined outputs.

Usage:
    python merge_encoding_results.py --results_dir results --output_dir .
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results",
                   help="Directory containing per-mouse .pkl files")
    p.add_argument("--output_dir",  default=".",
                   help="Where to write combined output files")
    return p.parse_args()


def main():
    args = parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.results_dir, "mouse_*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No mouse_*.pkl files found in {args.results_dir}")

    print(f"Merging {len(pkl_files)} result file(s) …")

    kernels_frames       = []
    reconstruction_dict  = {}
    reconstructed_frames = []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as fh:
            result = pickle.load(fh)

        mouse_id = result["mouse_id"]
        print(f"  {mouse_id}: {len(result['kernels_df'])} kernel rows")

        kernels_frames.append(result["kernels_df"])
        reconstruction_dict.update(result["reconstruction_dict"])
        reconstructed_frames.append(result["reconstructed_dff"])

    # ── combine kernels_df ───────────────────────────────────────────────────
    kernels_df = pd.concat(kernels_frames, ignore_index=True)

    # ── combine reconstructed_dff_df ─────────────────────────────────────────
    # Each per-mouse df contains only that mouse's sessions; concat by rows.
    reconstructed_dff_df = pd.concat(reconstructed_frames)
    # Drop any duplicate rows that might arise if session filtering overlapped.
    reconstructed_dff_df = reconstructed_dff_df[~reconstructed_dff_df.index.duplicated(keep="first")]

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    out_kernels   = os.path.join(args.output_dir, "kernels_df.pkl")
    out_recon_dat = os.path.join(args.output_dir, "reconstruction_data.pkl")
    out_recon_dff = os.path.join(args.output_dir, "reconstructed_dff_df.pkl")

    with open(out_kernels, "wb") as fh:
        pickle.dump(kernels_df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_recon_dat, "wb") as fh:
        pickle.dump(reconstruction_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_recon_dff, "wb") as fh:
        pickle.dump(reconstructed_dff_df, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDone.")
    print(f"  kernels_df          → {out_kernels}  ({len(kernels_df)} rows, {kernels_df['session_id'].nunique()} mice)")
    print(f"  reconstruction_data → {out_recon_dat}  ({len(reconstruction_dict)} entries)")
    print(f"  reconstructed_dff   → {out_recon_dff}  ({len(reconstructed_dff_df)} sessions)")


if __name__ == "__main__":
    main()
