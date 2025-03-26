# Create log directory
LOG_DIR="bias_evaluation/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

# Set maximum number of parallel tasks - increase to 12
MAX_PARALLEL=12

# Model list: deepseek-r1, deepseek-v3, gpt-4o, o3-mini
MODELS=("deepseek-r1" "deepseek-v3" "gpt-4o" "o3-mini" "deepseek-r1-distill-llama-70b" "llama-3.3-70b-instruct")
# MODELS=("llama-3.3-70b-instruct")
# Evaluation script list
EVALUATIONS=("psychology_authority_evaluation.py")


run_with_limit() {
    local limit=$1
    local count=0
    
    # Create a temporary FIFO file for inter-process communication
    temp_fifo="/tmp/eval_fifo_$$"
    mkfifo $temp_fifo
    
    # Start background process to read from the FIFO
    exec 3<>$temp_fifo
    rm $temp_fifo
    
    # Initialize FIFO with available slots
    for ((i=0; i<limit; i++)); do
        echo >&3
    done
    
    # Run task queue
    for task in "${@:2}"; do
        # Read one slot from the FIFO, indicating an available slot
        read -u 3
        
        # Execute task
        {
            eval "$task"
            # Upon completion, write back to FIFO to release slot
            echo >&3
        } &
    done
    
    # Wait for all tasks to complete
    wait
    
    # Close FIFO
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

# Run tasks with parallelism limit
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
