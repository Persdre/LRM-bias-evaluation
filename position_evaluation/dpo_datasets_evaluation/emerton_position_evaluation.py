#!/usr/bin/env python3
# Testing model's susceptibility to position bias when evaluating Emerton responses

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
                base_url="https://api.chatfire.cn/v1"):
    """Set up and return the OpenAI client."""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    return client

# Load Emerton DPO dataset from JSON file
def load_dataset(file_path, num_samples=100):
    """Load the Emerton DPO dataset from a JSON file and limit to num_samples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate prompt with chosen answer in position A
def generate_position_a_prompt(item):
    """Generate a prompt with chosen answer in position A."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    first_answer = item["chosen"]
    second_answer = item["rejected"]
    correct_option = "A"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer is better?\n\n"
        f"only respond with the letter (A or B)"
    )
    
    return system_prompt, formatted_prompt, correct_option

# Generate prompt with chosen answer in position B
def generate_position_b_prompt(item):
    """Generate a prompt with chosen answer in position B."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    first_answer = item["rejected"]
    second_answer = item["chosen"]
    correct_option = "B"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer is better?\n\n"
        f"only respond with the letter (A or B)"
    )
    
    return system_prompt, formatted_prompt, correct_option

# Get model response
def get_model_response(client, system_prompt, prompt, model_name, temperature=0.7, max_tokens=1024):
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

# Generate prompt in regular ordering (random positions)
def generate_regular_prompt(item):
    """Generate prompt in regular ordering."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    # Randomly decide which answer goes where
    if random.choice([True, False]):
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
    else:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer is better?\n\n"
        f"only respond with the letter (A or B)"
    )
    
    return system_prompt, formatted_prompt, correct_option

# Main evaluation function
def evaluate_position_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to position bias."""
    results = []
    
    # Track metrics
    position_a_correct = 0     # Correct when chosen answer in position A
    position_b_correct = 0     # Correct when chosen answer in position B
    regular_correct = 0        # Correct in regular ordering (random positions)
    position_a_count = 0       # Total valid evaluations with chosen answer in position A
    position_b_count = 0       # Total valid evaluations with chosen answer in position B
    regular_count = 0          # Total valid evaluations in regular ordering
    
    # Position preference tracking
    position_a_choices = 0     # Times model chose option A
    position_b_choices = 0     # Times model chose option B
    
    # IMPORTANT: This is the key metric we're focusing on
    # Track if model makes the same choice whether correct answer is at A or random position
    regular_vs_position_a_consistent = 0  # Times model made same choice between regular and all-in-A
    regular_vs_position_b_consistent = 0  # Times model made same choice between regular and all-in-B
    
    valid_pairs = 0            # Total valid evaluation pairs
    
    # Process each dataset item
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        # Skip invalid items
        if not all(k in item for k in ["input", "chosen", "rejected", "system"]):
            print(f"Skipping item {i} - missing required fields")
            continue
        
        try:
            # Get baseline (regular ordering) evaluation
            system_prompt, regular_prompt, regular_correct_option = generate_regular_prompt(item)
            regular_response = get_model_response(client, system_prompt, regular_prompt, model_name)
            
            if not regular_response:
                print(f"Skipping item {i} - no regular response")
                continue
                
            regular_choice = extract_answer_choice(regular_response)
            if not regular_choice:
                print(f"Skipping item {i} - couldn't extract regular choice")
                continue
                
            # Check if regular choice is correct
            regular_is_correct = regular_choice == regular_correct_option
            if regular_is_correct:
                regular_correct += 1
            regular_count += 1
            
            # In regular condition, track which position (A or B) was chosen
            if regular_choice == "A":
                position_a_choices += 1
            elif regular_choice == "B":
                position_b_choices += 1
            
            # Now test with chosen answer always in position A
            system_prompt_a, position_a_prompt, position_a_correct_option = generate_position_a_prompt(item)
            position_a_response = get_model_response(client, system_prompt_a, position_a_prompt, model_name)
            
            if not position_a_response:
                print(f"Skipping item {i} - no position A response")
                continue
                
            position_a_choice = extract_answer_choice(position_a_response)
            if not position_a_choice:
                print(f"Skipping item {i} - couldn't extract position A choice")
                continue
                
            # Check if position A choice is correct
            if position_a_choice == position_a_correct_option:
                position_a_correct += 1
            position_a_count += 1
            
            # In position A condition, also track which position was chosen
            if position_a_choice == "A":
                position_a_choices += 1
            elif position_a_choice == "B":
                position_b_choices += 1
            
            # Test with chosen answer in position B
            system_prompt_b, position_b_prompt, position_b_correct_option = generate_position_b_prompt(item)
            position_b_response = get_model_response(client, system_prompt_b, position_b_prompt, model_name)
            
            if not position_b_response:
                print(f"Skipping item {i} - no position B response")
                continue
                
            position_b_choice = extract_answer_choice(position_b_response)
            if not position_b_choice:
                print(f"Skipping item {i} - couldn't extract position B choice")
                continue
                
            # Check if position B choice is correct
            if position_b_choice == position_b_correct_option:
                position_b_correct += 1
            position_b_count += 1
            
            # In position B condition, also track which position was chosen
            if position_b_choice == "A":
                position_a_choices += 1
            elif position_b_choice == "B":
                position_b_choices += 1
            
            # KEY METRIC: Check if model made consistent choice regardless of position
            # For this to be true, we need to check if model chose the same *answer* (not position)
            # If in regular, the correct answer was at position A and model chose A, 
            # and in position_a, model chose A - that's consistent (always chose correct)
            # If in regular, the correct answer was at position B and model chose B,
            # and in position_a, model chose A - that's also consistent (always chose correct)
            
            # First determine what answer was chosen in regular condition
            chose_correct_in_regular = regular_choice == regular_correct_option
            
            # Then check if same answer was chosen in position_a
            chose_correct_in_position_a = position_a_choice == position_a_correct_option
            
            # If both are same, decision was consistent
            if chose_correct_in_regular == chose_correct_in_position_a:
                regular_vs_position_a_consistent += 1
            
            # Then check if same answer was chosen in position_b
            chose_correct_in_position_b = position_b_choice == position_b_correct_option
            
            # If both are same, decision was consistent
            if chose_correct_in_regular == chose_correct_in_position_b:
                regular_vs_position_b_consistent += 1
            
            # Count valid pair
            valid_pairs += 1
            
            # Store detailed result
            result = {
                "item_id": i,
                "input": item.get("input", ""),
                "chosen_answer": item.get("chosen", ""),
                "rejected_answer": item.get("rejected", ""),
                
                "regular_prompt": regular_prompt,
                "regular_response": regular_response,
                "regular_choice": regular_choice,
                "regular_correct_option": regular_correct_option,
                "regular_is_correct": regular_is_correct,
                
                "position_a_prompt": position_a_prompt,
                "position_a_response": position_a_response,
                "position_a_choice": position_a_choice,
                "position_a_correct_option": position_a_correct_option,
                "position_a_is_correct": position_a_choice == position_a_correct_option,
                
                "position_b_prompt": position_b_prompt,
                "position_b_response": position_b_response,
                "position_b_choice": position_b_choice,
                "position_b_correct_option": position_b_correct_option,
                "position_b_is_correct": position_b_choice == position_b_correct_option,
                
                # This is our key robustness metric
                "regular_vs_position_a_consistent": chose_correct_in_regular == chose_correct_in_position_a,
                "regular_vs_position_b_consistent": chose_correct_in_regular == chose_correct_in_position_b,
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
            
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    if valid_pairs == 0:
        print("No valid evaluation pairs completed!")
        return {"error": "No valid evaluations"}, []
    
    # Accuracy metrics
    regular_accuracy = regular_correct / regular_count if regular_count > 0 else 0
    position_a_accuracy = position_a_correct / position_a_count if position_a_count > 0 else 0
    position_b_accuracy = position_b_correct / position_b_count if position_b_count > 0 else 0
    
    # MAIN ROBUSTNESS METRIC: Consistency between regular and position A conditions
    regular_vs_position_a_robust_rate = regular_vs_position_a_consistent / valid_pairs
    
    # MAIN ROBUSTNESS METRIC: Consistency between regular and position B conditions
    regular_vs_position_b_robust_rate = regular_vs_position_b_consistent / valid_pairs
    
    # Position preference
    total_choices = position_a_choices + position_b_choices
    position_a_preference = position_a_choices / total_choices if total_choices > 0 else 0
    position_b_preference = position_b_choices / total_choices if total_choices > 0 else 0
    
    # Overall position bias
    position_bias = position_a_preference - 0.5  # >0 means bias towards A, <0 means bias towards B
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": len(results),
        "valid_pairs": valid_pairs,
        
        # Accuracy metrics
        "regular_accuracy": regular_accuracy,
        "position_a_accuracy": position_a_accuracy,
        "position_b_accuracy": position_b_accuracy,
        "accuracy_difference": position_a_accuracy - position_b_accuracy,
        
        # Position preference
        "position_a_preference": position_a_preference,
        "position_b_preference": position_b_preference,
        "position_bias": position_bias,
        
        # Robustness metric
        "regular_vs_position_a_robust_rate": regular_vs_position_a_robust_rate,
        "regular_vs_position_b_robust_rate": regular_vs_position_b_robust_rate,
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return summary, results

def get_default_output_filename(model_name, dataset_path):
    """Generate a default output filename based on model and dataset names."""
    # Extract dataset name from path
    # dataset is always emerton_dpo
    dataset_name = "emerton_dpo"
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_position_bias_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/position_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to position bias when evaluating Emerton responses')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/emerton_dpo_samples.json',
                        help='Path to Emerton DPO dataset JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for evaluation results')
    parser.add_argument('--model', type=str, default='deepseek-v3',
                        help='Model name to use for evaluation')
    parser.add_argument('--api_key', type=str, default="YOUR_OPENAI_APIKEY",
                        help='API key for the model service')
    parser.add_argument('--api_base', type=str, default="https://api.chatfire.cn/v1",
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
    print(f"Valid Pairs: {summary['valid_pairs']}")
    
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