# Create log directory
LOG_DIR="bias_evaluation/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

# Set max number of parallel tasks - increased to 24
MAX_PARALLEL=24

# List of models: deepseek-r1, deepseek-v3, gpt-4o, o3-mini
MODELS=("deepseek-r1" "deepseek-v3" "gpt-4o" "o3-mini" "deepseek-r1-distill-llama-70b" "llama-3.3-70b-instruct")
# MODELS=("llama-3.3-70b-instruct")
# List of evaluation scripts
EVALUATIONS=("math_bandwagon_evaluation.py" "chemistry_bandwagon_evaluation.py" "history_bandwagon_evaluation.py" "psychology_bandwagon_evaluation.py")

# Task management function - use process substitution instead of wait -n
run_with_limit() {
    local limit=$1
    local count=0
    
    # Create temporary named pipe for inter-process communication
    temp_fifo="/tmp/eval_fifo_$$"
    mkfifo $temp_fifo
    
    # Start background process and read from the pipe
    exec 3<>$temp_fifo
    rm $temp_fifo
    
    # Initialize pipe content
    for ((i=0; i<limit; i++)); do
        echo >&3
    done
    
    # Run the task queue
    for task in "${@:2}"; do
        # Read one item from the pipe, equivalent to acquiring an available slot
        read -u 3
        
        # Execute the task
        {
            eval "$task"
            # After the task finishes, write back to the pipe to release the slot
            echo >&3
        } &
    done
    
    # Wait for all tasks to complete
    wait
    
    # Close the pipe
    exec 3>&-
}

# Build the task list
TASKS=()
for eval_script in "${EVALUATIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Create a unique log filename
        log_file="${LOG_DIR}/$(basename $eval_script .py)_${model}.log"
        
        echo "Preparing: python $eval_script --model $model --samples 100 (Log: $log_file)"
        
        # Add to the task list
        TASKS+=("python $eval_script --model $model --samples 100 > $log_file 2>&1")
    done
done

# Run the tasks, limiting parallelism
echo "Starting evaluations with max $MAX_PARALLEL parallel tasks..."
run_with_limit $MAX_PARALLEL "${TASKS[@]}"

# Check results
echo "All evaluations completed. Check logs in $LOG_DIR"
echo "Summary of results:"
grep -r "accuracy" $LOG_DIR | sort

# JSON result processing
echo -e "\nJSON Results:"
RESULTS_DIR="$(dirname $(dirname $LOG_DIR))/results"
if [ -d "$RESULTS_DIR" ]; then
    # List recently modified JSON files
    find $RESULTS_DIR -name "*.json" -type f -mmin -60 | sort
    
    # Extract JSON result summaries
    echo -e "\nResults Summary:"
    for json_file in $(find $RESULTS_DIR -name "*.json" -type f -mmin -60); do
        echo "$(basename $json_file): $(grep -o '"accuracy":[^,}]*' $json_file | head -1)"
    done
fi
