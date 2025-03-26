# Create log directory
LOG_DIR="bias_evaluation/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

# Set the maximum number of parallel tasks
MAX_PARALLEL=8

# List of models
MODELS=("deepseek-r1" "deepseek-r1-distill-llama-70b" "gpt-4o")
# MODELS=("llama-3.3-70b-instruct")

# List of evaluation scripts
EVALUATIONS=("py_position_evaluation.py" "truthful_position_evaluation.py")

# Task management function - use process substitution instead of wait -n
run_with_limit() {
    local limit=$1
    local count=0
    
    # Create a temporary FIFO file for inter-process communication
    temp_fifo="/tmp/eval_fifo_$$"
    mkfifo $temp_fifo
    
    # Start background process to read from the pipe
    exec 3<>$temp_fifo
    rm $temp_fifo
    
    # Initialize pipe content
    for ((i=0; i<limit; i++)); do
        echo >&3
    done
    
    # Run task queue
    for task in "${@:2}"; do
        # Read one item from the pipe, equivalent to acquiring a slot
        read -u 3
        
        # Execute the task
        {
            eval "$task"
            # After task is done, write back to pipe to release the slot
            echo >&3
        } &
    done
    
    # Wait for all tasks to complete
    wait
    
    # Close the pipe
    exec 3>&-
}

# Build task list
TASKS=()
for eval_script in "${EVALUATIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Create a unique log file name
        log_file="${LOG_DIR}/$(basename $eval_script .py)_${model}.log"
        
        echo "Preparing: python $eval_script --model $model --samples 100 (Log: $log_file)"
        
        # Add to task list
        TASKS+=("python $eval_script --model $model --samples 100 > $log_file 2>&1")
    done
done

# Run tasks with limited parallelism
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
    
    # Extract JSON result summary
    echo -e "\nResults Summary:"
    for json_file in $(find $RESULTS_DIR -name "*.json" -type f -mmin -60); do
        echo "$(basename $json_file): $(grep -o '"accuracy":[^,}]*' $json_file | head -1)"
    done
fi

# Print execution statistics
echo -e "\nExecution Statistics:"
echo "Total evaluations: ${#EVALUATIONS[@]}"
echo "Total models: ${#MODELS[@]}"
echo "Total tasks: ${#TASKS[@]}"
echo "Execution time: $SECONDS seconds"
