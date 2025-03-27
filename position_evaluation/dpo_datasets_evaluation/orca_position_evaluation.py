#!/usr/bin/env python3
# Testing model's susceptibility to position bias when evaluating Orca responses

import json
import argparse
import os
import time
from tqdm import tqdm
import random
from openai import OpenAI
import re
from typing import List, Dict, Any
from datetime import datetime

# Set up the OpenAI client with your provided API credentials
def setup_client(api_key="YOUR_OPENAI_APIKEY", 
                base_url="https://api.openai.com/v1"):
    """Set up and return the OpenAI client."""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    return client

# Load dataset from JSON file
def load_dataset(file_path, num_samples=100):
    """Load the dataset from a JSON file and limit to num_samples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate prompt with chosen answer in position A
def generate_position_a_prompt(item):
    """Generate a prompt with chosen answer in position A."""
    prompt = item["prompt"]
    
    first_answer = item["chosen"]
    second_answer = item["rejected"]
    correct_option = "A"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer (A or B) is better?\n\n"
        f"Respond only with just the letter (A or B)."
    )
    
    return formatted_prompt, correct_option

# Generate prompt with chosen answer in position B
def generate_position_b_prompt(item):
    """Generate a prompt with chosen answer in position B."""
    prompt = item["prompt"]
    
    first_answer = item["rejected"]
    second_answer = item["chosen"]
    correct_option = "B"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer (A or B) is better?\n\n"
        f"Respond only with just the letter (A or B)."
    )
    
    return formatted_prompt, correct_option

# Generate prompt with regular ordering (baseline)
def generate_regular_prompt(item):
    """Generate a regular prompt with randomly positioned answers."""
    prompt = item["prompt"]
    
    # Randomly decide which answer goes where for the baseline test
    is_swapped = random.choice([True, False])
    
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer (A or B) is better?\n\n"
        f"Respond only with just the letter (A or B)."
    )
    
    return formatted_prompt, correct_option

# Get model response
def get_model_response(client, prompt, model_name, system_prompt, temperature=0.7, max_tokens=1024):
    """Get response from the model."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting model response: {e}")
        time.sleep(5)  # Wait before retrying
        return None

# Extract the answer choice (A or B) from the model's response
def extract_answer_choice(response):
    """Extract the answer choice (A or B) from the model's response."""
    if not response:
        return None
    
    # Check for answers like "The answer is A" or just "A"
    patterns = [
        r"(?:answer is|choice is|select|choose|option is|answer:|choice:|option:)\s*([AB])",
        r"(?:answer:|i choose|i select|final answer:|therefore:|thus:|conclusion:|in conclusion:).*?([AB])(?:\.|$)",
        r"\b([AB])\b"  # Look for a standalone A or B
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()  # Return the first match uppercase
    
    return None

# Main evaluation function
def evaluate_position_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to position bias when choosing Orca responses."""
    results = []
    
    # Track metrics
    position_a_correct = 0
    position_b_correct = 0
    regular_correct = 0  # New: track regular condition accuracy
    
    # Count position choices
    position_a_choices = 0
    position_b_choices = 0
    
    # Track consistency
    regular_vs_position_a_robust = 0  # New: track robustness between regular and position A
    regular_vs_position_b_robust = 0  # New: track robustness between regular and position B
    valid_evaluations = 0
    
    # System prompt
    system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    # Process each item
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        try:
            # New: First test with regular ordering (baseline)
            regular_prompt, regular_correct_option = generate_regular_prompt(item)
            regular_response = get_model_response(client, regular_prompt, model_name, system_prompt)
            regular_choice = extract_answer_choice(regular_response)
            
            # Check if regular choice is valid
            if not regular_choice:
                print(f"Skipping item {i} - could not extract regular choice")
                continue
                
            # Determine which answer was chosen in regular condition
            regular_chose_chosen = (regular_choice == regular_correct_option)
            if regular_chose_chosen:
                regular_correct += 1
            
            # First test: Chosen answer in position A
            position_a_prompt, position_a_correct_option = generate_position_a_prompt(item)
            position_a_response = get_model_response(client, position_a_prompt, model_name, system_prompt)
            position_a_choice = extract_answer_choice(position_a_response)
            
            # Check if position A choice is valid
            if not position_a_choice:
                print(f"Skipping item {i} - could not extract position A choice")
                continue
                
            # Track position A metrics
            position_a_chose_chosen = (position_a_choice == position_a_correct_option)
            if position_a_chose_chosen:
                position_a_correct += 1
                
            if position_a_choice == "A":
                position_a_choices += 1
            elif position_a_choice == "B":
                position_b_choices += 1
            
            # New: Check robustness between regular and position A
            if regular_chose_chosen == position_a_chose_chosen:
                regular_vs_position_a_robust += 1
            
            # Second test: Chosen answer in position B
            position_b_prompt, position_b_correct_option = generate_position_b_prompt(item)
            position_b_response = get_model_response(client, position_b_prompt, model_name, system_prompt)
            position_b_choice = extract_answer_choice(position_b_response)
            
            # Check if position B choice is valid
            if not position_b_choice:
                print(f"Skipping item {i} - could not extract position B choice")
                continue
                
            # Track position B metrics
            position_b_chose_chosen = (position_b_choice == position_b_correct_option)
            if position_b_chose_chosen:
                position_b_correct += 1
            
            # New: Check robustness between regular and position B
            if regular_chose_chosen == position_b_chose_chosen:
                regular_vs_position_b_robust += 1
            
            # Store results
            result = {
                "id": item.get("id", f"sample_{i}"),
                "question": item["prompt"],
                "chosen_answer": item["chosen"],
                "rejected_answer": item["rejected"],
                
                "regular_prompt": regular_prompt,
                "regular_response": regular_response,
                "regular_choice": regular_choice,
                "regular_correct_option": regular_correct_option,
                "regular_is_correct": regular_chose_chosen,
                
                "position_a_prompt": position_a_prompt,
                "position_a_response": position_a_response,
                "position_a_choice": position_a_choice,
                "position_a_correct_option": position_a_correct_option,
                "position_a_is_correct": position_a_chose_chosen,
                
                "position_b_prompt": position_b_prompt,
                "position_b_response": position_b_response,
                "position_b_choice": position_b_choice,
                "position_b_correct_option": position_b_correct_option,
                "position_b_is_correct": position_b_chose_chosen,
                
                "regular_vs_position_a_robust": regular_chose_chosen == position_a_chose_chosen,
                "regular_vs_position_b_robust": regular_chose_chosen == position_b_chose_chosen
            }
            
            results.append(result)
            valid_evaluations += 1
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
            
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    total = valid_evaluations
    if total == 0:
        print("No valid evaluations completed!")
        return {"error": "No valid evaluations"}, []
        
    regular_accuracy = regular_correct / total
    position_a_accuracy = position_a_correct / total
    position_b_accuracy = position_b_correct / total
    
    total_choices = position_a_choices + position_b_choices
    position_a_preference = position_a_choices / total_choices if total_choices > 0 else 0
    position_b_preference = position_b_choices / total_choices if total_choices > 0 else 0
    
    # Position bias (deviation from 50/50)
    position_bias = position_a_preference - 0.5
    
    # Calculate robustness metrics
    regular_vs_position_a_robust_rate = regular_vs_position_a_robust / total
    regular_vs_position_b_robust_rate = regular_vs_position_b_robust / total
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "position_a_accuracy": position_a_accuracy,
        "position_b_accuracy": position_b_accuracy,
        "accuracy_difference": position_a_accuracy - position_b_accuracy,
        "position_a_preference": position_a_preference,
        "position_b_preference": position_b_preference,
        "position_bias": position_bias,
        "regular_vs_position_a_robust_rate": regular_vs_position_a_robust_rate,
        "regular_vs_position_b_robust_rate": regular_vs_position_b_robust_rate
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return summary, results

def get_default_output_filename(model_name, dataset_path):
    """Generate a default output filename based on model and dataset names."""
    # Extract dataset name from path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_position_bias_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/position_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to position bias when evaluating Orca responses')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/orca_dpo_samples.json',
                        help='Path to Orca DPO dataset JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for evaluation results')
    parser.add_argument('--model', type=str, default='deepseek-v3',
                        help='Model name to use for evaluation')
    parser.add_argument('--api_key', type=str, default="YOUR_OPENAI_APIKEY",
                        help='API key for the model service')
    parser.add_argument('--api_base', type=str, default="https://api.openai.com/v1",
                        help='Base URL for the API')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to evaluate (default: 100)')
    
    args = parser.parse_args()
    
    # Generate default output filename if none provided
    if args.output is None:
        args.output = get_default_output_filename(args.model, args.dataset)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(args.output)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Set up client
    client = setup_client(args.api_key, args.api_base)
    
    # Load dataset with sample limit
    dataset = load_dataset(args.dataset, args.samples)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Evaluate
    summary, _ = evaluate_position_bias(
        client, 
        dataset, 
        args.model, 
        args.output
    )
    
    # Print summary
    print("\nPosition Bias Evaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    
    print("\n*** POSITION BIAS METRICS ***")
    print(f"Regular Accuracy (Baseline): {summary['regular_accuracy']:.2%}")
    print(f"Position A Accuracy: {summary['position_a_accuracy']:.2%}")
    print(f"Position B Accuracy: {summary['position_b_accuracy']:.2%}")
    
    print("\nRobustness Metrics:")
    print(f"Regular vs Position A Robust Rate: {summary['regular_vs_position_a_robust_rate']:.2%}")
    print(f"Regular vs Position B Robust Rate: {summary['regular_vs_position_b_robust_rate']:.2%}")
    print("(This measures how often the model makes the same choice regardless of position)")
    
    print("\nPosition Preference Analysis:")
    print(f"Option A Preference: {summary['position_a_preference']:.2%}")
    print(f"Option B Preference: {summary['position_b_preference']:.2%}")
    bias_direction = "towards first position (A)" if summary['position_bias'] > 0 else "towards second position (B)"
    print(f"Position Bias: {abs(summary['position_bias']):.2%} {bias_direction} (0% = unbiased)")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()