#!/usr/bin/env python3
# Testing model's susceptibility to position bias when answering psychology questions
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
    """Load the psychology dataset from a JSON file and limit to num_samples if specified."""
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
def get_model_response(client, prompt, model_name, temperature=0.0, max_tokens=20):
    """Get response from the model."""
    try:
        # Set up system prompt for psychology questions
        system_prompt = "You are a helpful assistant that can answer psychology questions accurately."
        
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
        time.sleep(2)  # Wait before retrying
        return None

# Extract the answer choice from the model's response
def extract_answer_choice(response):
    """Extract the answer choice (letter) from the model's response."""
    if not response:
        return None
    
    # Try to find a standalone letter (A, B, C, D, etc.)
    letter_pattern = r'\b([A-Z])\b'
    matches = re.findall(letter_pattern, response.upper())
    if matches:
        return matches[0]
    
    # Try other common patterns
    patterns = [
        r'(?:answer is|choice is|select|choose|option is|answer:|choice:|option:)\s*([A-Z])',
        r'(?:answer:|i choose|i select|final answer:|therefore:|thus:|conclusion:|in conclusion:).*?([A-Z])(?:\.|$)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()  # Return the first match uppercase
    
    return None

# Main evaluation function
def evaluate_position_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to position bias in psychology questions."""
    results = []
    
    # Track metrics
    total_questions = len(dataset)
    valid_evaluated_questions = 0
    
    # Regular condition (baseline)
    regular_correct = 0
    regular_count = 0
    regular_choices = {}  # Track distribution of choices in regular condition
    
    # Position First condition (correct answer in position A)
    position_first_correct = 0
    position_first_count = 0
    position_first_choices = {}  # Track distribution of choices in position A condition
    
    # Position Last condition (correct answer in last position)
    position_last_correct = 0
    position_last_count = 0
    position_last_choices = {}  # Track distribution of choices in position Last condition
    
    # Consistency tracking
    regular_correct_responses = []  # To track if model was correct in regular condition
    position_first_correct_responses = []  # To track if model was correct in position A condition
    position_last_correct_responses = []  # To track if model was correct in position Last condition
    
    last_letters = []  # Track what the last letter was for each question
    
    # Process each dataset item
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name} on psychology questions")):
        result = {
            "index": i,
            "question": item["question"] if "question" in item else "N/A",
            "original_answer": item["answer"] if "answer" in item else "N/A",
        }
        
        # Skip items without required fields
        if not all(k in item for k in ["question", "options", "answer"]):
            print(f"Skipping item {i} - missing required fields")
            result["skipped"] = True
            results.append(result)
            continue
        
        # Test with regular prompt (baseline)
        regular_prompt, regular_correct_answer = generate_regular_prompt(item)
        regular_response = get_model_response(client, regular_prompt, model_name)
        
        if not regular_response:
            print(f"Skipping item {i} - no regular response")
            result["skipped"] = True
            results.append(result)
            continue
            
        regular_choice = extract_answer_choice(regular_response)
        
        if not regular_choice:
            print(f"Skipping item {i} - failed to extract regular choice")
            result["skipped"] = True
            results.append(result)
            continue
            
        # Count choice distribution
        regular_choices[regular_choice] = regular_choices.get(regular_choice, 0) + 1
        
        # Check if regular choice is correct
        regular_is_correct = (regular_choice == regular_correct_answer)
        regular_correct_responses.append(regular_is_correct)
        if regular_is_correct:
            regular_correct += 1
        regular_count += 1
        
        # Test with correct answer in position A (first)
        position_first_prompt, position_first_correct_answer = generate_position_first_prompt(item)
        position_first_response = get_model_response(client, position_first_prompt, model_name)
        
        if not position_first_response:
            print(f"Skipping item {i} - no position A response")
            result["skipped"] = True
            results.append(result)
            continue
            
        position_first_choice = extract_answer_choice(position_first_response)
        
        if not position_first_choice:
            print(f"Skipping item {i} - failed to extract position A choice")
            result["skipped"] = True
            results.append(result)
            continue
            
        # Count choice distribution
        position_first_choices[position_first_choice] = position_first_choices.get(position_first_choice, 0) + 1
        
        # Check if position first choice is correct
        position_first_is_correct = (position_first_choice == position_first_correct_answer)
        position_first_correct_responses.append(position_first_is_correct)
        if position_first_is_correct:
            position_first_correct += 1
        position_first_count += 1
        
        # Test with correct answer in last position
        position_last_prompt, position_last_correct_answer = generate_position_last_prompt(item)
        position_last_response = get_model_response(client, position_last_prompt, model_name)
        
        if not position_last_response:
            print(f"Skipping item {i} - no position Last response")
            result["skipped"] = True
            results.append(result)
            continue
            
        position_last_choice = extract_answer_choice(position_last_response)
        
        if not position_last_choice:
            print(f"Skipping item {i} - failed to extract position Last choice")
            result["skipped"] = True
            results.append(result)
            continue
            
        # Track what the last letter was
        last_letters.append(position_last_correct_answer)
        
        # Count choice distribution
        position_last_choices[position_last_choice] = position_last_choices.get(position_last_choice, 0) + 1
        
        # Check if position last choice is correct
        position_last_is_correct = (position_last_choice == position_last_correct_answer)
        position_last_correct_responses.append(position_last_is_correct)
        if position_last_is_correct:
            position_last_correct += 1
        position_last_count += 1
        
        # Only increment valid questions if all three conditions were evaluated
        valid_evaluated_questions += 1
        
        # Store results for this item
        result.update({
            "skipped": False,
            "regular_prompt": regular_prompt,
            "regular_response": regular_response,
            "regular_choice": regular_choice,
            "regular_correct_answer": regular_correct_answer,
            "regular_is_correct": regular_is_correct,
            "position_first_prompt": position_first_prompt,
            "position_first_response": position_first_response,
            "position_first_choice": position_first_choice,
            "position_first_correct_answer": position_first_correct_answer,
            "position_first_is_correct": position_first_is_correct,
            "position_last_prompt": position_last_prompt,
            "position_last_response": position_last_response,
            "position_last_choice": position_last_choice,
            "position_last_correct_answer": position_last_correct_answer,
            "position_last_is_correct": position_last_is_correct
        })
        
        results.append(result)
    
    # Calculate metrics
    regular_accuracy = regular_correct / regular_count if regular_count > 0 else 0
    position_first_accuracy = position_first_correct / position_first_count if position_first_count > 0 else 0
    position_last_accuracy = position_last_correct / position_last_count if position_last_count > 0 else 0
    
    # Calculate position effects (compared to regular)
    position_first_effect = position_first_accuracy - regular_accuracy
    position_last_effect = position_last_accuracy - regular_accuracy
    
    # Calculate difference between position first and position last
    first_vs_last_accuracy_diff = position_first_accuracy - position_last_accuracy
    
    # Calculate consistency rates
    regular_vs_position_first_matches = sum(1 for r, a in zip(regular_correct_responses, position_first_correct_responses) if r == a)
    regular_vs_position_last_matches = sum(1 for r, l in zip(regular_correct_responses, position_last_correct_responses) if r == l)
    position_first_vs_last_matches = sum(1 for a, l in zip(position_first_correct_responses, position_last_correct_responses) if a == l)
    
    total_pairs = len(regular_correct_responses)
    
    regular_vs_position_first_robust_rate = regular_vs_position_first_matches / total_pairs if total_pairs > 0 else 0
    regular_vs_position_last_robust_rate = regular_vs_position_last_matches / total_pairs if total_pairs > 0 else 0
    position_first_vs_last_robust_rate = position_first_vs_last_matches / total_pairs if total_pairs > 0 else 0
    
    # Calculate position preference rates
    position_first_rate = position_first_choices.get("A", 0) / position_first_count if position_first_count > 0 else 0
    
    # Find most common last letter
    if last_letters:
        from collections import Counter
        letter_counts = Counter(last_letters)
        most_common_last_letter = letter_counts.most_common(1)[0][0]
    else:
        most_common_last_letter = "D"  # Default if we can't determine
        
    position_last_rate = position_last_choices.get(most_common_last_letter, 0) / position_last_count if position_last_count > 0 else 0
    
    # Create summary
    summary = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "valid_evaluated_questions": valid_evaluated_questions,
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
    dataset_name = "psychology_samples"
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_first_vs_last_position_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/position_evaluation/fact_datasets_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to position bias (first vs last) on psychology questions')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/mmlu_pro_samples/psychology_samples.json',
                        help='Path to psychology dataset JSON')
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