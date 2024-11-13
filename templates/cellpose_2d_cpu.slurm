#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=cellpose
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output ../tests/output/log/%x_%j.out

INPUT_FOLDER="../tests/data/merfish"
OUTPUT_FOLDER="../tests/output/cellpose_2d"
OUTPUT_FILE="../tests/output/cellpose_2d.json"

# Get experiment FOVs
FOVS=$(fishtank fovs --input $INPUT_FOLDER)
IFS=',' read -r -a FOV_LIST <<< "$FOVS"

# Run cellpose on each FOV in parallel
for FOV in "${FOV_LIST[@]}"; do
    srun -N1 -n1 --exclusive fishtank cellpose \
        --fov ${FOV} \
        --input $INPUT_FOLDER \
        --output $OUTPUT_FOLDER \
        --file_pattern {series}/Conv_zscan_{fov}.dax \
        --model nuclei \
        --diameter 80 \
        --channels DAPI \
        --min_size 100 \
        --z_slices 4 &
done

wait

# Aggregate polygons
fishtank aggregate-polygons \
    --input $OUTPUT_FOLDER \
    --output $OUTPUT_FILE \
    --min_size 10 \
    --min_ioa .2
