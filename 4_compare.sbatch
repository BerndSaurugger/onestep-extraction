#!/bin/bash
#SBATCH --job-name=gather_groundtruths
#SBATCH --output=logs/gather_%j.out
#SBATCH --error=logs/gather_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=GPU-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

mkdir -p output_parquets

# 2. Analyse starten
python gather_groundtruth_labels.py \
    --gen_folder sdv3_bb_synthall \
    --parquet_file output_parquets/bb_attack_top400.parquet \
    --out_parquet_file output_parquets/bb_attack_sdv1_multiple_full_analyzed.parquet \
    --download_reals True