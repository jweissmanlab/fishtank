#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=cellpose
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --partition=nvidia-t4-20,nvidia-A6000-20
#SBATCH --gres=gpu:1
#SBATCH --account=weissman
#SBATCH --output ../tests/output/log/%x_%j.out

INPUT_FOLDER="../tests/data/merfish"
OUTPUT_FOLDER="../tests/output/cellpose_3d"

# Get experiment FOVs
FOVS=$(fishtank fovs --input $INPUT_FOLDER)
IFS=',' read -r -a FOV_LIST <<< "$FOVS"

for FOV in "${FOV_LIST[@]}"; do
    srun -n1 -N1 --mem  16G --exclusive fishtank cellpose \
        --fov ${FOV} \
        --input $INPUT_FOLDER \
        --output $OUTPUT_FOLDER \
        --file_pattern {series}/Conv_zscan_{fov}.dax \
        --model nuclei \
        --diameter 17 \
        --channels DAPI \
        --downsample 4 \
        --do_3D True \
        --filter deconwolf \
        --gpu True &
done

wait
