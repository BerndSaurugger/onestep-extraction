#!/bin/bash
#SBATCH --job-name=bb_attack_synthall
#SBATCH --output=logs/bb_attack_synthall_%A_%a.out
#SBATCH --error=logs/bb_attack_synthall_%A_%a.err
#SBATCH --array=0-3                # 4 jobs parallel
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=GPU-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00   # länger wegen mehr steps

# Anzahl der Captions als Hyperparameter
AMOUNT=400

TOTAL_CAPTIONS=$(python -c "from utils.functions import best_scoring; df=best_scoring(filename='output_parquets/bb_attack_sdv1_30k_full_analyzed', amount=$AMOUNT, drop=False); print(len(df))")

N_JOBS=4
N_CAPTIONS_PER_JOB=$(( (TOTAL_CAPTIONS + N_JOBS - 1) / N_JOBS ))  # ceil div
OFFSET=$(( SLURM_ARRAY_TASK_ID * N_CAPTIONS_PER_JOB ))

python prepare_for_multiple.py --input_file output_parquets/bb_attack_sdv1_30k_full_analyzed.parquet --output_file output_parquets/bb_attack_top${AMOUNT}.parquet --amount $AMOUNT

python synthall_from_parquet.py \
    --model="runwayml/stable-diffusion-v1-5" \
    --outfolder=sdv3_bb_synthall/part${SLURM_ARRAY_TASK_ID}/ \
    --parquet_file=output_parquets/bb_attack_top${AMOUNT}.parquet \
    --caption_offset=${OFFSET} \
    --n_captions=${N_CAPTIONS_PER_JOB} \
    --steps=32 \
    --n_seeds=4

for i in {0..3}; do
    PART_DIR=sdv3_bb_synthall/part$i
    if [ -d "$PART_DIR" ]; then
        for sub in "$PART_DIR"/*; do
            BASENAME=$(basename "$sub")
            DEST=sdv3_bb_synthall/"$BASENAME"
            if [ -e "$DEST" ]; then
                echo "Ordner $DEST existiert bereits, überspringe Verschiebung von $sub"
            else
                mv "$sub" sdv3_bb_synthall/
            fi
        done
        rmdir "$PART_DIR" 2>/dev/null || echo "Ordner $PART_DIR konnte nicht gelöscht werden (nicht leer?)"
    else
        echo "Ordner $PART_DIR existiert nicht."
    fi
done
