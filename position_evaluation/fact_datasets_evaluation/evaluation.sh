

LOG_DIR="bias_evaluation/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"


MAX_PARALLEL=24

MODELS=("deepseek-r1" "deepseek-r1-distill-llama-70b" "gpt-4o")
# MODELS=("llama-3.3-70b-instruct")


EVALUATIONS=("chemistry_position_evaluation.py" "history_position_evaluation.py" "psychology_position_evaluation.py" "math_position_evaluation.py")

run_with_limit() {
    local limit=$1
    local count=0
    
    temp_fifo="/tmp/eval_fifo_$$"
    mkfifo $temp_fifo
    

    exec 3<>$temp_fifo
    rm $temp_fifo
    

    for ((i=0; i<limit; i++)); do
        echo >&3
    done
    

    for task in "${@:2}"; do

        read -u 3
        

        {
            eval "$task"

            echo >&3
        } &
    done
    

    wait
    
    exec 3>&-
}


TASKS=()
for eval_script in "${EVALUATIONS[@]}"; do
    for model in "${MODELS[@]}"; do

        log_file="${LOG_DIR}/$(basename $eval_script .py)_${model}.log"
        
        echo "Preparing: python $eval_script --model $model --samples 100 (Log: $log_file)"
        

        TASKS+=("python $eval_script --model $model --samples 100 > $log_file 2>&1")
    done
done

=
echo "Starting evaluations with max $MAX_PARALLEL parallel tasks..."
run_with_limit $MAX_PARALLEL "${TASKS[@]}"


echo "All evaluations completed. Check logs in $LOG_DIR"
echo "Summary of results:"
grep -r "accuracy" $LOG_DIR | sort


echo -e "\nJSON Results:"
RESULTS_DIR="$(dirname $(dirname $LOG_DIR))/results"
if [ -d "$RESULTS_DIR" ]; then

    find $RESULTS_DIR -name "*.json" -type f -mmin -60 | sort
    

    echo -e "\nResults Summary:"
    for json_file in $(find $RESULTS_DIR -name "*.json" -type f -mmin -60); do
        echo "$(basename $json_file): $(grep -o '"accuracy":[^,}]*' $json_file | head -1)"
    done
fi


echo -e "\nExecution Statistics:"
echo "Total evaluations: ${#EVALUATIONS[@]}"
echo "Total models: ${#MODELS[@]}"
echo "Total tasks: ${#TASKS[@]}"
echo "Execution time: $SECONDS seconds"
