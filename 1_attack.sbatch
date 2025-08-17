#!/bin/bash
#SBATCH --job-name=bb_attack
#SBATCH --output=logs/bb_attack_%A_%a.out
#SBATCH --error=logs/bb_attack_%A_%a.err
#SBATCH --array=0-3                # 4 jobs in parallel
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=GPU-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00


# Parameters
TOTAL_CAPTIONS=$(python -c "import pandas as pd; d=pd.read_parquet('membership_attack_top30k.parquet'); print(len(d))")
N_JOBS=4  # or $SLURM_ARRAY_TASK_COUNT
N_CAPTIONS_PER_JOB=$(( (TOTAL_CAPTIONS + N_JOBS - 1) / N_JOBS ))  # ceil div
OFFSET=$(( SLURM_ARRAY_TASK_ID * N_CAPTIONS_PER_JOB ))
python run_bb_attack.py \
    --out_parquet_file=bb_attack_sdv1_30k_part${SLURM_ARRAY_TASK_ID}.parquet \
    --caption_offset=${OFFSET} \
    --n_captions=${N_CAPTIONS_PER_JOB} \
    --outfolder=gen_onestep/sdv1_bb_attack/part${SLURM_ARRAY_TASK_ID}/ \
    --n_seeds=4

for i in {0..3}; do
    mv gen_onestep/sdv1_bb_attack/part$i/* gen_onestep/sdv1_bb_attack/
done
rmdir gen_onestep/sdv1_bb_attack/part*
