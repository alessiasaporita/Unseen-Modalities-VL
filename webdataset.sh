#!/bin/bash
#SBATCH --job-name=food
#SBATCH --output=/work/tesi_asaporita//UnseenModalities-VL/Output/food101_training_%a
#SBATCH --error=/work/tesi_asaporita//UnseenModalities-VL/Output/food101_training_%a
#SBATCH --partition=all_usr_prod
#SBATCH --account=tesi_asaporita
#SBATCH --mem=20G
#SBATCH --array=0-20
#SBATCH --time=5:00:00
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_A40_48G"



. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate uns_modalities
cd /work/tesi_asaporita/UnseenModalities-VL

srun --exclusive python -u web_dataset.py --split training --text_path  /work/tesi_asaporita/checkpoint/Hatefull_Memes/text/checkpoint/HM_text_7.pt --image_path /work/tesi_asaporita/checkpoint/Hatefull_Memes/image/checkpoint/HM_image_5.pt --job_idx $SLURM_ARRAY_TASK_ID --job_size 3000 &

wait
