#!/usr/bin/env python3
# Testing model's susceptibility to position bias when answering history questions
# comparing first position vs. last position

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
def load_dataset(file_path, num_samples=None):
    """Load the history dataset from a JSON file and limit to num_samples if specified."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit to specified number of samples if provided
    if num_samples is not None:
        return data[:num_samples]
    return data

# Generate prompt with the correct answer in position A (first position)
def generate_position_first_prompt(item):
    """Generate a prompt with the correct answer in first position (A)."""
    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    
    # Find the index of the correct answer
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    correct_index = option_letters.index(correct_answer_letter)
    
    # Rearrange options to put correct answer first (position A)
    rearranged_options = [options[correct_index]]  # Start with correct answer
    for i in range(len(options)):
        if i != correct_index:
            rearranged_options.append(options[i])  # Add other options
    
    # Format the rearranged options with letters
    formatted_options = ""
    for i, option in enumerate(rearranged_options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    # New correct answer is always 'A'
    new_correct_answer = "A"
    
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, new_correct_answer

# Generate prompt with the correct answer in last position
def generate_position_last_prompt(item):
    """Generate a prompt with the correct answer in last position."""
    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    
    # Find the index of the correct answer
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    correct_index = option_letters.index(correct_answer_letter)
    
    # Rearrange options to put correct answer last
    rearranged_options = []
    
    # Add all incorrect options first
    for i in range(len(options)):
        if i != correct_index:
            rearranged_options.append(options[i])
            
    # Add correct answer at the end
    rearranged_options.append(options[correct_index])
    
    # Format the rearranged options with letters
    formatted_options = ""
    for i, option in enumerate(rearranged_options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    # New correct answer is the last letter
    new_correct_answer = option_letters[len(options)-1]
    
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, new_correct_answer

# Generate prompt with regular ordering (baseline)
def generate_regular_prompt(item):
    """Generate a regular prompt without rearranging options."""
    question = item["question"]
    options = item["options"]
    correct_answer = item["answer"]
    
    # Get option letters based on number of options
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    
    # Format the options with letters
    formatted_options = ""
    for i, option in enumerate(options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    # Construct the prompt
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, correct_answer

# Get model response
def get_model_response(client, prompt, model_name, system_prompt="You are a helpful assistant that can answer history questions accurately.", temperature=0.0, max_tokens=1024):
    """Get model response for a given prompt."""
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

# Extract answer choice from model response
def extract_answer_choice(response):
    """Extract the answer choice (letter) from the model's response."""
    if not response:
        return None
    
    # Look for a letter at the beginning of the response
    match = re.search(r"^\s*([A-Z])", response)
    if match:
        return match.group(1)
    
    # Look for phrases like "the answer is X" or "I choose X"
    match = re.search(r"(?:answer|choice|option)(?:\s+is)?\s+([A-Z])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Just look for any standalone letter
    match = re.search(r"\b([A-Z])\b", response)
    if match:
        return match.group(1)
    
    return None

# Main evaluation function
def evaluate_position_bias(client, dataset, model_name, output_file):
    """Evaluate position bias by comparing performance when correct answer is in first vs last position."""
    results = []
    
    # Initialize counters
    total_questions = 0
    regular_correct = 0
    position_first_correct = 0
    position_last_correct = 0
    
    # Track option choices in each condition
    regular_choices = {}
    position_first_choices = {}
    position_last_choices = {}
    
    # Track consistency between conditions
    regular_vs_position_first_robust = 0
    regular_vs_position_last_robust = 0
    position_first_vs_last_robust = 0
    
    # Count different last letters
    last_letters_counts = {}
    
    # System prompt
    system_prompt = "You are a helpful assistant that can answer history questions accurately."
    
    # Process each item in the dataset
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        try:
            # Test 1: Regular ordering (baseline)
            regular_prompt, regular_correct_answer = generate_regular_prompt(item)
            regular_response = get_model_response(client, regular_prompt, model_name, system_prompt)
            regular_choice = extract_answer_choice(regular_response)
            
            if not regular_choice:
                print(f"Warning: Could not extract answer choice from regular response for item {i}")
                continue
                
            regular_is_correct = (regular_choice == regular_correct_answer)
            
            # Update regular choices counter
            regular_choices[regular_choice] = regular_choices.get(regular_choice, 0) + 1
            
            # Test 2: Correct answer in first position
            position_first_prompt, position_first_correct_answer = generate_position_first_prompt(item)
            position_first_response = get_model_response(client, position_first_prompt, model_name, system_prompt)
            position_first_choice = extract_answer_choice(position_first_response)
            
            if not position_first_choice:
                print(f"Warning: Could not extract answer choice from position first response for item {i}")
                continue
                
            position_first_is_correct = (position_first_choice == position_first_correct_answer)
            
            # Update first position choices counter
            position_first_choices[position_first_choice] = position_first_choices.get(position_first_choice, 0) + 1
            
            # Test 3: Correct answer in last position
            position_last_prompt, position_last_correct_answer = generate_position_last_prompt(item)
            position_last_response = get_model_response(client, position_last_prompt, model_name, system_prompt)
            position_last_choice = extract_answer_choice(position_last_response)
            
            if not position_last_choice:
                print(f"Warning: Could not extract answer choice from position last response for item {i}")
                continue
                
            position_last_is_correct = (position_last_choice == position_last_correct_answer)
            
            # Update last position choices counter
            position_last_choices[position_last_choice] = position_last_choices.get(position_last_choice, 0) + 1
            
            # Track the last option letter frequency
            last_letter = position_last_correct_answer
            last_letters_counts[last_letter] = last_letters_counts.get(last_letter, 0) + 1
            
            # Update counters
            total_questions += 1
            if regular_is_correct:
                regular_correct += 1
            if position_first_is_correct:
                position_first_correct += 1
            if position_last_is_correct:
                position_last_correct += 1
                
            # Check robustness (consistency across conditions)
            if regular_is_correct == position_first_is_correct:
                regular_vs_position_first_robust += 1
            if regular_is_correct == position_last_is_correct:
                regular_vs_position_last_robust += 1
            if position_first_is_correct == position_last_is_correct:
                position_first_vs_last_robust += 1
            
            # Store result
            result = {
                "id": i,
                "question": item["question"],
                "original_options": item["options"],
                "original_correct_answer": item["answer"],
                
                "regular_response": regular_response,
                "regular_choice": regular_choice,
                "regular_is_correct": regular_is_correct,
                
                "position_first_response": position_first_response,
                "position_first_choice": position_first_choice,
                "position_first_is_correct": position_first_is_correct,
                
                "position_last_response": position_last_response,
                "position_last_choice": position_last_choice,
                "position_last_is_correct": position_last_is_correct,
                
                "position_last_letter": position_last_correct_answer
            }
            results.append(result)
            
            # Avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    
    # Calculate metrics
    if total_questions > 0:
        regular_accuracy = regular_correct / total_questions
        position_first_accuracy = position_first_correct / total_questions
        position_last_accuracy = position_last_correct / total_questions
        
        # Position effects (change in accuracy when positions are manipulated)
        position_first_effect = position_first_accuracy - regular_accuracy
        position_last_effect = position_last_accuracy - regular_accuracy
        
        # Position preference (how often each position is chosen)
        position_first_rate = position_first_choices.get("A", 0) / total_questions
        
        # Find most common last letter
        most_common_last_letter = max(last_letters_counts, key=last_letters_counts.get)
        position_last_rate = position_last_choices.get(most_common_last_letter, 0) / total_questions
        
        # Consistency rates
        regular_vs_position_first_robust_rate = regular_vs_position_first_robust / total_questions
        regular_vs_position_last_robust_rate = regular_vs_position_last_robust / total_questions
        position_first_vs_last_robust_rate = position_first_vs_last_robust / total_questions
    else:
        regular_accuracy = position_first_accuracy = position_last_accuracy = 0
        position_first_effect = position_last_effect = 0
        position_first_rate = position_last_rate = 0
        regular_vs_position_first_robust_rate = regular_vs_position_last_robust_rate = position_first_vs_last_robust_rate = 0
    
    # Calculate position bias metrics
    first_vs_last_accuracy_diff = position_first_accuracy - position_last_accuracy
    
    # Create summary
    summary = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "regular_accuracy": regular_accuracy,
        "position_first_accuracy": position_first_accuracy,
        "position_last_accuracy": position_last_accuracy,
        "first_vs_last_accuracy_diff": first_vs_last_accuracy_diff,
        "position_first_effect": position_first_effect,
        "position_last_effect": position_last_effect,
        "position_first_rate": position_first_rate,
        "position_last_rate": position_last_rate,
        "most_common_last_letter": most_common_last_letter,
        "position_first_choices": position_first_choices,
        "position_last_choices": position_last_choices,
        "regular_choices": regular_choices,
        "regular_vs_position_first_robust_rate": regular_vs_position_first_robust_rate,
        "regular_vs_position_last_robust_rate": regular_vs_position_last_robust_rate,
        "position_first_vs_last_robust_rate": position_first_vs_last_robust_rate
    }
    
    # Save results to file
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
    dataset_name = "history_samples"
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_first_vs_last_position_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/position_evaluation/fact_datasets_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to position bias (first vs last) on history questions')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/mmlu_pro_samples/history_samples.json',
                        help='Path to history dataset JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for evaluation results')
    parser.add_argument('--model', type=str, default='deepseek-v3',
                        help='Model name to use for evaluation')
    parser.add_argument('--api_key', type=str, default="YOUR_OPENAI_APIKEY",
                        help='API key for the model service')
    parser.add_argument('--api_base', type=str, default="https://api.openai.com/v1",
                        help='Base URL for the API')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    
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
    print("\nPosition Bias Evaluation Summary (First vs Last):")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    
    print("\n*** ACCURACY BY POSITION ***")
    print(f"Regular (unchanged) accuracy: {summary['regular_accuracy']:.2%}")
    print(f"First Position (correct answer first) accuracy: {summary['position_first_accuracy']:.2%}")
    print(f"Last Position (correct answer last) accuracy: {summary['position_last_accuracy']:.2%}")
    print(f"First vs Last accuracy difference: {summary['first_vs_last_accuracy_diff']*100:+.2f} pp")
    
    print("\n*** POSITION EFFECTS ***")
    print(f"First Position effect (vs regular): {summary['position_first_effect']*100:+.2f} pp")
    print(f"Last Position effect (vs regular): {summary['position_last_effect']*100:+.2f} pp")
    
    print("\n*** POSITION PREFERENCE ***")
    print(f"Selection rate for option A when correct answer is first: {summary['position_first_rate']:.2%}")
    print(f"Selection rate for option {summary['most_common_last_letter']} when correct answer is last: {summary['position_last_rate']:.2%}")
    
    print("\n*** CHOICE DISTRIBUTION ***")
    print("Regular condition choices:", {k: v for k, v in summary["regular_choices"].items() if v > 0})
    print("First Position condition choices:", {k: v for k, v in summary["position_first_choices"].items() if v > 0})
    print("Last Position condition choices:", {k: v for k, v in summary["position_last_choices"].items() if v > 0})
    
    print("\n*** CONSISTENCY METRICS ***")
    print(f"Regular vs First Position robust rate: {summary['regular_vs_position_first_robust_rate']:.2%}")
    print(f"Regular vs Last Position robust rate: {summary['regular_vs_position_last_robust_rate']:.2%}")
    print(f"First vs Last Position robust rate: {summary['position_first_vs_last_robust_rate']:.2%}")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main() 