#!/usr/bin/env python3
# Testing model's susceptibility to position bias when answering math questions
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
    """Load the math dataset from a JSON file and limit to num_samples if specified."""
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
def get_model_response(client, prompt, model_name):
    """Get model response for a given prompt."""
    try:
        # System message for math questions
        system_message = "You are a helpful assistant that can answer math questions accurately."
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            timeout=30
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting model response: {e}")
        return None

# Extract answer choice from model response
def extract_answer_choice(response_text):
    """Extract the answer choice letter from the model's response."""
    if not response_text:
        return None
        
    # First try: look for single capital letter at beginning of response
    first_word = response_text.strip().split()[0] if response_text.strip() else ""
    if len(first_word) == 1 and first_word.isalpha() and first_word.isupper():
        return first_word
    
    # Second try: extract any single capital letter
    letters = re.findall(r'\b[A-Z]\b', response_text)
    if letters:
        return letters[0]
    
    # Third try: look for letter followed by period or parenthesis
    matches = re.findall(r'([A-Z])[.)]', response_text)
    if matches:
        return matches[0]
    
    # Fourth try: check for common patterns like "Option A" or "answer is B"
    option_pattern = re.search(r'[Oo]ption\s+([A-Z])', response_text)
    if option_pattern:
        return option_pattern.group(1)
        
    answer_pattern = re.search(r'[Aa]nswer\s+(?:is\s+)?([A-Z])', response_text)
    if answer_pattern:
        return answer_pattern.group(1)
    
    # Last attempt: just find any capital letter A-Z in the response
    capitals = re.findall(r'[A-Z]', response_text)
    if capitals:
        for capital in capitals:
            if capital in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return capital
    
    return None

# Main evaluation function
def evaluate_position_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to position bias."""
    results = []
    
    # Track metrics
    position_first_correct = 0     # Correct when answer in first position
    position_last_correct = 0      # Correct when answer in last position
    regular_correct = 0            # Correct in regular ordering
    position_first_count = 0       # Total valid evaluations with answer in first position
    position_last_count = 0        # Total valid evaluations with answer in last position
    regular_count = 0              # Total valid evaluations in regular ordering
    
    # Track choices by letter
    position_first_choices = {}    # Choices in first position condition
    position_last_choices = {}     # Choices in last position condition
    regular_choices = {}           # Choices in regular condition
    
    # Initialize counters for all possible letter choices
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        position_first_choices[letter] = 0
        position_last_choices[letter] = 0
        regular_choices[letter] = 0
    
    # Consistency tracking
    regular_vs_position_first_robust = 0   # Times model made same decision between regular and first
    regular_vs_position_last_robust = 0    # Times model made same decision between regular and last
    position_first_vs_last_robust = 0      # Times model made same decision between first and last
    
    total_questions = 0
    valid_evaluated_questions = 0
    
    # Process each dataset item
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        try:
            # Get baseline (regular ordering) evaluation
            regular_prompt, regular_correct_option = generate_regular_prompt(item)
            regular_response = get_model_response(client, regular_prompt, model_name)
            
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
            
            # In regular condition, track which position was chosen
            regular_choices[regular_choice] = regular_choices.get(regular_choice, 0) + 1
            
            # Now test with correct answer always in first position
            position_first_prompt, position_first_correct_option = generate_position_first_prompt(item)
            position_first_response = get_model_response(client, position_first_prompt, model_name)
            
            if not position_first_response:
                print(f"Skipping item {i} - no first position response")
                continue
                
            position_first_choice = extract_answer_choice(position_first_response)
            if not position_first_choice:
                print(f"Skipping item {i} - couldn't extract first position choice")
                continue
                
            # Check if first position choice is correct
            position_first_is_correct = position_first_choice == position_first_correct_option
            if position_first_is_correct:
                position_first_correct += 1
            position_first_count += 1
            
            # In first position condition, track which position was chosen
            position_first_choices[position_first_choice] = position_first_choices.get(position_first_choice, 0) + 1
            
            # Test with correct answer in last position
            position_last_prompt, position_last_correct_option = generate_position_last_prompt(item)
            position_last_response = get_model_response(client, position_last_prompt, model_name)
            
            if not position_last_response:
                print(f"Skipping item {i} - no last position response")
                continue
                
            position_last_choice = extract_answer_choice(position_last_response)
            if not position_last_choice:
                print(f"Skipping item {i} - couldn't extract last position choice")
                continue
                
            # Check if last position choice is correct
            position_last_is_correct = position_last_choice == position_last_correct_option
            if position_last_is_correct:
                position_last_correct += 1
            position_last_count += 1
            
            # In last position condition, track which position was chosen
            position_last_choices[position_last_choice] = position_last_choices.get(position_last_choice, 0) + 1
            
            # Check consistency across conditions
            # If model got it right (or wrong) in both conditions, it's consistent
            if regular_is_correct == position_first_is_correct:
                regular_vs_position_first_robust += 1
                
            if regular_is_correct == position_last_is_correct:
                regular_vs_position_last_robust += 1
                
            if position_first_is_correct == position_last_is_correct:
                position_first_vs_last_robust += 1
            
            # Increment total valid evaluations
            valid_evaluated_questions += 1
            
            # Store detailed result
            result = {
                "item_id": i,
                "question": item.get("question", ""),
                "original_options": item.get("options", []),
                "original_answer": item.get("answer", ""),
                
                "regular_prompt": regular_prompt,
                "regular_response": regular_response,
                "regular_choice": regular_choice,
                "regular_correct_option": regular_correct_option,
                "regular_is_correct": regular_is_correct,
                
                "position_first_prompt": position_first_prompt,
                "position_first_response": position_first_response,
                "position_first_choice": position_first_choice,
                "position_first_correct_option": position_first_correct_option,
                "position_first_is_correct": position_first_is_correct,
                
                "position_last_prompt": position_last_prompt,
                "position_last_response": position_last_response,
                "position_last_choice": position_last_choice,
                "position_last_correct_option": position_last_correct_option,
                "position_last_is_correct": position_last_is_correct,
                
                # Consistency metrics
                "regular_vs_position_first_consistent": regular_is_correct == position_first_is_correct,
                "regular_vs_position_last_consistent": regular_is_correct == position_last_is_correct,
                "position_first_vs_last_consistent": position_first_is_correct == position_last_is_correct,
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
            
        # Increment total questions counter
        total_questions += 1
            
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    if valid_evaluated_questions == 0:
        print("No valid evaluation pairs completed!")
        return {"error": "No valid evaluations"}, []
    
    # Accuracy metrics
    regular_accuracy = regular_correct / regular_count if regular_count > 0 else 0
    position_first_accuracy = position_first_correct / position_first_count if position_first_count > 0 else 0
    position_last_accuracy = position_last_correct / position_last_count if position_last_count > 0 else 0
    
    # Position effect (change from baseline)
    position_first_effect = position_first_accuracy - regular_accuracy
    position_last_effect = position_last_accuracy - regular_accuracy
    
    # Position bias metrics
    first_vs_last_accuracy_diff = position_first_accuracy - position_last_accuracy
    
    # Robustness rates
    regular_vs_position_first_robust_rate = regular_vs_position_first_robust / valid_evaluated_questions
    regular_vs_position_last_robust_rate = regular_vs_position_last_robust / valid_evaluated_questions
    position_first_vs_last_robust_rate = position_first_vs_last_robust / valid_evaluated_questions
    
    # Position preference analysis
    position_first_rate = position_first_choices.get("A", 0) / position_first_count if position_first_count > 0 else 0
    
    # Find the most common last letter across questions
    last_letters = [chr(ord('A') + len(item["options"]) - 1) for item in dataset if "options" in item]
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
    dataset_name = "math_samples"
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_first_vs_last_position_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/position_evaluation/fact_datasets_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to position bias (first vs last) on math questions')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/mmlu_pro_samples/math_samples.json',
                        help='Path to math dataset JSON')
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