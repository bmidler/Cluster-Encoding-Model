#!/bin/bash
# ============================================================
# submit_encoding_jobs.sh
#
# 1. Discovers all mice from the data pickles.
# 2. Writes mice_list.txt.
# 3. Submits a SLURM array job (one task per mouse).
# 4. After all array tasks finish, submits a merge job.
# 5. After merge finishes, submits a plotting job.
#
# Usage:
#   bash submit_encoding_jobs.sh
# ============================================================

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
PHOTOMETRY_PKL="dff_data.pkl"
ANNOTATIONS_PKL="segmentation_data.pkl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
MICE_LIST="${SCRIPT_DIR}/mice_list.txt"
LOGS_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# ── conda env ────────────────────────────────────────────────────────────────
CONDA_ENV="general"    # change if your env has a different name

# ── activate conda (needed for the discovery step below) ────────────────────
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# ── discover mice ────────────────────────────────────────────────────────────
echo "Discovering mice from data …"
python3 - <<'PYEOF'
import pickle, sys, os

session_type = "Defeat"   # must match what you pass to the worker

with open("dff_data.pkl", "rb") as fh:
    dff = pickle.load(fh)
with open("segmentation_data.pkl", "rb") as fh:
    seg = pickle.load(fh)

common = set(dff.index) & set(seg.index)
if session_type:
    common = {s for s in common if session_type in s}

mice = set()
for sid in common:
    parts = sid.split("-")
    mouse_id = parts[2] if len(parts) >= 3 else parts[0]
    mice.add(mouse_id)

mice_sorted = sorted(mice)
with open("mice_list.txt", "w") as fh:
    fh.write("\n".join(mice_sorted) + "\n")

print(f"Found {len(mice_sorted)} mice: {mice_sorted}")
PYEOF

N_MICE=$(wc -l < "${MICE_LIST}")
if [[ "${N_MICE}" -eq 0 ]]; then
    echo "ERROR: no mice found. Check your pickle files and session_type filter."
    exit 1
fi

LAST_IDX=$(( N_MICE - 1 ))
echo "Submitting array job for ${N_MICE} mice (indices 0–${LAST_IDX}) …"

# ── submit the array job ─────────────────────────────────────────────────────
ARRAY_JOB_ID=$(sbatch \
    --parsable \
    --array="0-${LAST_IDX}" \
    "${SCRIPT_DIR}/fit_encoding_array.sh")

echo "Array job submitted: ${ARRAY_JOB_ID}"

# ── submit the merge job, dependent on the array finishing ───────────────────
MERGE_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --job-name="EncodingMerge" \
    --partition=witten \
    --mem=128GB \
    --time=02:00:00 \
    --output="${LOGS_DIR}/merge-slurm-%j.txt" \
    --error="${LOGS_DIR}/merge-error-%j.txt" \
    --wrap="
        eval \"\$($HOME/miniconda3/bin/conda shell.bash hook)\"
        conda activate ${CONDA_ENV}
        python3 ${SCRIPT_DIR}/merge_encoding_results.py \
            --results_dir ${RESULTS_DIR} \
            --output_dir  ${SCRIPT_DIR}
    ")

echo "Merge job submitted: ${MERGE_JOB_ID} (runs after array job completes)"

# ── submit the plotting job, dependent on merge finishing ────────────────────
PLOT_JOB_ID=$(sbatch \
    --parsable \
    --dependency="afterok:${MERGE_JOB_ID}" \
    --job-name="EncodingPlots" \
    --partition=witten \
    --mem=128GB \
    --time=02:00:00 \
    --output="${LOGS_DIR}/plots-slurm-%j.txt" \
    --error="${LOGS_DIR}/plots-error-%j.txt" \
    --wrap="
        eval \"\$($HOME/miniconda3/bin/conda shell.bash hook)\"
        conda activate ${CONDA_ENV}
        python3 ${SCRIPT_DIR}/plot_encoding_results.py \
            --kernels_pkl    ${SCRIPT_DIR}/kernels_df.pkl \
            --recon_data_pkl ${SCRIPT_DIR}/reconstruction_data.pkl \
            --output_dir     ${SCRIPT_DIR}/plots
    ")

echo "Plot job submitted:  ${PLOT_JOB_ID} (runs after merge job completes)"
echo ""
echo "Pipeline:"
echo "  Array [${ARRAY_JOB_ID}] → Merge [${MERGE_JOB_ID}] → Plots [${PLOT_JOB_ID}]"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Array logs:    ${LOGS_DIR}/worker-<mouse_idx>-slurm-<job>.txt"
echo "Merge log:     ${LOGS_DIR}/merge-slurm-<job>.txt"
echo "Plot log:      ${LOGS_DIR}/plots-slurm-<job>.txt"
echo ""
echo "Outputs written to:"
echo "  ${SCRIPT_DIR}/plots/kernel_psth/        ← all-mice kernel PSTHs (2×3 grid)"
echo "  ${SCRIPT_DIR}/plots/kernel_psth_groups/ ← Resilient vs Susceptible PSTHs"
echo "  ${SCRIPT_DIR}/plots/cv_alpha/           ← per-mouse CV alpha vs MSE curves"
echo "  ${SCRIPT_DIR}/plots/photometry_psth/    ← PSTH original vs reconstructed"
echo "  ${SCRIPT_DIR}/plots/photometry_traces/  ← full time-series, per mouse"
