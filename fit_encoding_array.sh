#!/bin/bash
#================================================================
# Linear Encoding Model — SLURM array script
# One task per mouse; SLURM_ARRAY_TASK_ID indexes mice_list.txt.
#
# Submit via submit_encoding_jobs.sh, not directly.
#================================================================

#SBATCH --job-name=EncodingModel
#SBATCH --partition=witten
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=%x-logs/worker-%a-slurm-%j.txt
#SBATCH --error=%x-logs/worker-%a-error-%j.txt

# SLURM_SUBMIT_DIR is set by SLURM to the directory where sbatch was called.
# Never use BASH_SOURCE[0] in array scripts — it resolves to the spool copy.
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

mkdir -p "${SCRIPT_DIR}/logs"

module load anacondapy/2023.07-cuda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate general

echo "============================================"
echo "Job ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
date
echo "============================================"

python3 "${SCRIPT_DIR}/fit_encoding_worker.py" \
    --mice_list       "${SCRIPT_DIR}/mice_list.txt" \
    --task_id         "${SLURM_ARRAY_TASK_ID}" \
    --photometry_pkl  "${SCRIPT_DIR}/dff_data.pkl" \
    --annotations_pkl "${SCRIPT_DIR}/segmentation_data.pkl" \
    --output_dir      "${SCRIPT_DIR}/results" \
    --kernel_length_sec 10 \
    --pre_onset_sec     5 \
    --target_fps        8 \
    --regularization    ridge \
    --alpha_low         1 \
    --alpha_high        10 \
    --alpha_n           50 \
    --l1_ratio          0.5 \
    --session_type      Defeat
    # --offset          # uncomment to include offset kernels
    # --days_to_include 1 2 3   # uncomment to restrict days

echo ""
echo "Done."
date
