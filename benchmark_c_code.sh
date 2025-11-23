#!/bin/bash
#SBATCH --cpus-per-task=1              # Number of CPU cores allocated per task
#SBATCH --mem-per-cpu=128G             # Memory allocated per CPU core
#SBATCH --partition=ou_sloan_gpu       # SLURM partition with GPU access
#SBATCH --time=24:00:00                # Maximum allowed running time
#SBATCH --gres=gpu:h100:1              # Request 1 NVIDIA H100 GPU
#SBATCH --job-name=vtx-benchmark       # Name of the job shown in SLURM queue
#SBATCH --array=1-2                    # Run as an array job with two tasks (1=normal, 2=reversed order)
#SBATCH --mail-type=BEGIN,END,FAIL     # Send email on job start, end, and failure
#SBATCH --mail-user=botang@mit.edu     # Email address for SLURM notifications
#SBATCH --output=/home/botang/src/cuPDLPx/slurm/slurm-c-%x-%A_%a.out   # Output log file (%x=job name, %A job ID, %a array ID)

# default benchmark (mittelmann if not specified by user)
BENCH=${BENCH:-mittelmann}
echo "Running benchmark: $BENCH"

# dataset selection based on BENCH variable
if [[ "$BENCH" == "mittelmann" ]]; then
    DATASET_DIR="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/mittelmann"   # Folder containing Mittelmann LP instances
    INSTANCE_FILE="mittelmann_instance.txt"                                      # File listing Mittelmann instance names (1 column)
elif [[ "$BENCH" == "miplib" ]]; then
    DATASET_DIR="/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017"    # Folder containing MIPLIB 2017 LP instances
    INSTANCE_FILE="miplib_instance.txt"                                           # File listing MIPLIB instances (3 columns: name class nnz)
else
    echo "Unknown BENCH=$BENCH, must be 'mittelmann' or 'miplib'"
    exit 1
fi

# output folder with current date (YYYYMMDD)
TODAY=$(date +%Y%m%d)
OUTPUT_DIR="/home/botang/src/cuPDLPx/result/${BENCH}_${TODAY}"
mkdir -p "$OUTPUT_DIR"

# load instance list from file into array
mapfile -t instance_list < "$INSTANCE_FILE"

# array job #2 runs the list in reverse order
if [[ "$SLURM_ARRAY_TASK_ID" -eq 2 ]]; then
    mapfile -t instance_list < <(printf '%s\n' "${instance_list[@]}" | tac)
fi


echo "Loaded ${#instance_list[@]} instances from ${INSTANCE_FILE}"

# iterate over all instances
for line in "${instance_list[@]}"; do
    if [[ "$BENCH" == "mittelmann" ]]; then
        instance_name="$line"   # Mittelmann: instance name only
        cls=""                  # no class label
        nnz=""                  # no nnz information
    else
        read -r instance_name cls nnz <<< "$line"   # MIPLIB: three fields (name class nnz)
    fi

    # construct paths
    instance_path="${DATASET_DIR}/${instance_name}.mps.gz"      # input MPS file
    summary_file="${OUTPUT_DIR}/${instance_name}.txt"           # output summary file

    echo "Instance: $instance_name  | Class: ${cls:-N/A}"
    echo "Summary file: $summary_file"

    # skip instance if output already exists
    if [ -f "$summary_file" ]; then
        echo "Skipping instance (already solved): $instance_path"
        continue
    fi

    # assign time limit
    if [[ "$BENCH" == "mittelmann" ]]; then
        TIMELIMIT=15000                       # Mittelmann: fixed 15,000 seconds
    else
        case "$cls" in
            Small|small)   TIMELIMIT=3600  ;; # Small MIPLIB: 3,600 seconds
            Medium|medium) TIMELIMIT=3600  ;; # Medium MIPLIB: 3,600 seconds
            Large|large)   TIMELIMIT=18000 ;; # Large MIPLIB: 18,000 seconds
        esac
    fi
    echo "Using timelimit = ${TIMELIMIT}s"

    # run solver
    /home/botang/src/cuPDLPx/build/cupdlpx --time_limit "${TIMELIMIT}" "${instance_path}" "${OUTPUT_DIR}"

    echo "Finished: $instance_path"
    echo "--------------------------------------"
done

echo "Benchmarking completed! Results saved in $OUTPUT_DIR"
