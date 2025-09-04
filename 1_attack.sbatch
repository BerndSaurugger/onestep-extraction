#!/bin/bash
#SBATCH --job-name=bb_attack
#SBATCH --output=logs/bb_attack_%A_%a.out
#SBATCH --error=logs/bb_attack_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=GPU-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# --- Verzeichnisse ---
mkdir -p gen_onestep/sdv1_bb_attack
mkdir -p output_parquets

# --- Chunking berechnen ---
TOTAL_CAPTIONS=$(python -c "import pandas as pd; d=pd.read_parquet('membership_attack_top30k.parquet'); print(len(d))")
N_JOBS=$SLURM_ARRAY_TASK_COUNT
N_CAPTIONS_PER_JOB=$(( (TOTAL_CAPTIONS + N_JOBS - 1) / N_JOBS ))
OFFSET=$(( SLURM_ARRAY_TASK_ID * N_CAPTIONS_PER_JOB ))

OUT_FOLDER="gen_onestep/sdv1_bb_attack/part${SLURM_ARRAY_TASK_ID}/"
mkdir -p $OUT_FOLDER

# --- BB-Attack ausführen ---
python run_bb_attack.py \
    --out_parquet_file=output_parquets/bb_attack_sdv1_30k_part${SLURM_ARRAY_TASK_ID}.parquet \
    --caption_offset=${OFFSET} \
    --n_captions=${N_CAPTIONS_PER_JOB} \
    --outfolder=${OUT_FOLDER} \
    --n_seeds=10

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Waiting for all array jobs to finish..."
    # ⚠ Nur mit Slurm >=23.02
    scontrol wait_job $SLURM_ARRAY_JOB_ID  

    # --- Bilder zusammenführen ---
    MERGE_DIR="gen_onestep/sdv1_bb_attack/"
    for part in part0 part1 part2 part3; do
        if [ -d "${MERGE_DIR}${part}" ]; then
            echo "Merging ${part}..."
            mv ${MERGE_DIR}${part}/* ${MERGE_DIR}/
            rmdir ${MERGE_DIR}${part}
        fi
    done

    # --- Parquets zusammenführen ---
    python - <<EOF
import pandas as pd
import glob

parquets = sorted(glob.glob("output_parquets/bb_attack_sdv1_30k_part*.parquet"))
dfs = [pd.read_parquet(p) for p in parquets]
full_df = pd.concat(dfs, ignore_index=True)
full_df.to_parquet("output_parquets/bb_attack_sdv1_30k_full.parquet")
print("Merged Parquet saved: output_parquets/bb_attack_sdv1_30k_full.parquet")
EOF
fi
