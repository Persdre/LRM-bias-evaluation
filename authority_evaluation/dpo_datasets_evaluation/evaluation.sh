#!/bin/bash

# Create log directory
LOG_DIR="bias_evaluation/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

# Set maximum number of parallel tasks
MAX_PARALLEL=12

# Task management function - use process substitution instead of wait -n
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

# Construct the task list of original evaluation commands
TASKS=(
    # # Original model evaluation commands
    # "python emerton_authority_evaluation.py --model gpt-4o --samples 100 > ${LOG_DIR}/emerton_gpt-4o.log 2>&1"
    # "python emerton_authority_evaluation.py --model o3-mini --samples 100 > ${LOG_DIR}/emerton_o3-mini.log 2>&1"
    
    # "python orca_authority_evaluation.py --model gpt-4o --samples 100 > ${LOG_DIR}/orca_gpt-4o.log 2>&1" 
    # "python orca_authority_evaluation.py --model o3-mini --samples 100 > ${LOG_DIR}/orca_o3-mini.log 2>&1"
    
    # "python py_authority_evaluation.py --model gpt-4o --samples 100 > ${LOG_DIR}/py_gpt-4o.log 2>&1"
    # "python py_authority_evaluation.py --model o3-mini --samples 100 > ${LOG_DIR}/py_o3-mini.log 2>&1"
    
    # "python truthful_authority_evaluation.py --model gpt-4o --samples 100 > ${LOG_DIR}/truthful_gpt-4o.log 2>&1"
    # "python truthful_authority_evaluation.py --model o3-mini --samples 100 > ${LOG_DIR}/truthful_o3-mini.log 2>&1"
    
    # # Add DeepSeek model evaluation commands
    # "python emerton_authority_evaluation.py --model deepseek-r1 --samples 100 > ${LOG_DIR}/emerton_deepseek-r1.log 2>&1"
    # "python emerton_authority_evaluation.py --model deepseek-v3 --samples 100 > ${LOG_DIR}/emerton_deepseek-v3.log 2>&1"
    
    # "python orca_authority_evaluation.py --model deepseek-r1 --samples 100 > ${LOG_DIR}/orca_deepseek-r1.log 2>&1"
    # "python orca_authority_evaluation.py --model deepseek-v3 --samples 100 > ${LOG_DIR}/orca_deepseek-v3.log 2>&1"
    
    # "python py_authority_evaluation.py --model deepseek-r1 --samples 100 > ${LOG_DIR}/py_deepseek-r1.log 2>&1"
    # "python py_authority_evaluation.py --model deepseek-v3 --samples 100 > ${LOG_DIR}/py_deepseek-v3.log 2>&1"
    
    # "python truthful_authority_evaluation.py --model deepseek-r1 --samples 100 > ${LOG_DIR}/truthful_deepseek-r1.log 2>&1"
    # "python truthful_authority_evaluation.py --model deepseek-v3 --samples 100 > ${LOG_DIR}/truthful_deepseek-v3.log 2>&1"

    # # Add llama-3.3-70b-instruct
    # "python emerton_authority_evaluation.py --model llama-3.3-70b-instruct --samples 100 > ${LOG_DIR}/emerton_llama-3.3-70b-instruct.log 2>&1"
    # "python orca_authority_evaluation.py --model llama-3.3-70b-instruct --samples 100 > ${LOG_DIR}/orca_llama-3.3-70b-instruct.log 2>&1"
    # "python py_authority_evaluation.py --model llama-3.3-70b-instruct --samples 100 > ${LOG_DIR}/py_llama-3.3-70b-instruct.log 2>&1"
    # "python truthful_authority_evaluation.py --model llama-3.3-70b-instruct --samples 100 > ${LOG_DIR}/truthful_llama-3.3-70b-instruct.log 2>&1"

    # Add deepseek-r1-distill-llama-70b
    "python emerton_authority_evaluation.py --model deepseek-r1-distill-llama-70b --samples 100 > ${LOG_DIR}/emerton_deepseek-r1-distill-llama-70b.log 2>&1"
    "python orca_authority_evaluation.py --model deepseek-r1-distill-llama-70b --samples 100 > ${LOG_DIR}/orca_deepseek-r1-distill-llama-70b.log 2>&1"
    "python py_authority_evaluation.py --model deepseek-r1-distill-llama-70b --samples 100 > ${LOG_DIR}/py_deepseek-r1-distill-llama-70b.log 2>&1"
    "python truthful_authority_evaluation.py --model deepseek-r1-distill-llama-70b --samples 100 > ${LOG_DIR}/truthful_deepseek-r1-distill-llama-70b.log 2>&1"
)

# Execute tasks with parallelism limit
echo "Starting evaluations with max $MAX_PARALLEL parallel tasks..."
run_with_limit $MAX_PARALLEL "${TASKS[@]}"

# Check results
echo "All evaluations completed. Check logs in $LOG_DIR"
echo "Summary of results:"
grep -r "accuracy\|score\|result" $LOG_DIR | sort

# JSON result processing
echo -e "\nJSON Results:"
RESULTS_DIR="$(dirname $(dirname $LOG_DIR))/results"
if [ -d "$RESULTS_DIR" ]; then
    # List recently modified JSON files
    find $RESULTS_DIR -name "*.json" -type f -mmin -60 | sort
    
    # Extract JSON result summaries
    echo -e "\nResults Summary:"
    for json_file in $(find $RESULTS_DIR -name "*.json" -type f -mmin -60); do
        echo "$(basename $json_file): $(grep -o '\"accuracy\":[^,}]*\|\"score\":[^,}]*' $json_file | head -1)"
    done
fi

# Execution statistics
echo -e "\nExecution Statistics:"
echo "Total tasks executed: ${#TASKS[@]}"
echo "Execution time: $SECONDS seconds"
