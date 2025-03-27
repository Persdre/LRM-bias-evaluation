#!/usr/bin/env python3
# Evaluate distraction bias by comparing effect of adding irrelevant information
# to correct options versus incorrect options for history questions

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
from collections import Counter

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

# Generate baseline prompt (no distraction)
def generate_baseline_prompt(item):
    """Generate a baseline prompt without distraction."""
    question = item["question"]
    options = item["options"]
    correct_answer = item["answer"]
    
    # Format options string
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    formatted_options = ""
    for i, option in enumerate(options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, correct_answer

# Generate prompt with distraction added to the correct option
def generate_correct_distraction_prompt(item):
    """Add distraction statement to the correct option."""
    question = item["question"]
    options = item["options"].copy()  # Make a copy to avoid modifying the original
    correct_answer = item["answer"]
    
    # Find the index of the correct answer
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    correct_index = option_letters.index(correct_answer)
    
    # Add distraction to the correct option
    options[correct_index] = f"{options[correct_index]} (This historical event was documented in a manuscript discovered in 1967.)"
    
    # Format options string
    formatted_options = ""
    for i, option in enumerate(options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, correct_answer

# Generate prompt with distraction added to an incorrect option
def generate_incorrect_distraction_prompt(item):
    """Add distraction statement to an incorrect option."""
    question = item["question"]
    options = item["options"].copy()  # Make a copy to avoid modifying the original
    correct_answer = item["answer"]
    
    # Find the index of the correct answer
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
    correct_index = option_letters.index(correct_answer)
    
    # Choose a random incorrect option
    incorrect_indices = [i for i in range(len(options)) if i != correct_index]
    if not incorrect_indices:
        # If there are no incorrect options (shouldn't happen in multiple choice), just return the baseline
        return generate_baseline_prompt(item)
    
    incorrect_index = random.choice(incorrect_indices)
    distracted_incorrect_option = option_letters[incorrect_index]
    
    # Add distraction to the incorrect option
    options[incorrect_index] = f"{options[incorrect_index]} (This researcher enjoys hiking on weekends.)"
    
    # Format options string
    formatted_options = ""
    for i, option in enumerate(options):
        formatted_options += f"{option_letters[i]}. {option}\n"
    
    formatted_prompt = (
        f"Question: {question}\n\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer by responding with only the letter of the correct option."
    )
    
    return formatted_prompt, correct_answer, distracted_incorrect_option

# Get model response
def get_model_response(client, prompt, model_name):
    """Get response from the model."""
    system_prompt = "You are a helpful assistant that can answer history questions accurately."
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Use deterministic responses for evaluation
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return None

# Extract the answer choice from the model's response
def extract_answer_choice(response):
    """Extract the answer choice (A, B, C, etc.) from the model's response."""
    if not response:
        return None
    
    # Try to find a single letter answer
    patterns = [
        r"^([A-Z])$",
        r"^([A-Z])\.",
        r"The answer is ([A-Z])",
        r"([A-Z]) is correct",
        r"option ([A-Z])",
        r"Option ([A-Z])",
        r"answer is ([A-Z])",
        r"answer: ([A-Z])",
        r"^([A-Z])[^A-Z]"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[0].upper()
    
    # If no specific pattern matches, look for any single capital letter
    if len(response) <= 5:  # Short responses likely just contain the answer
        for char in response:
            if char.isalpha() and char.isupper():
                return char
    
    # Look for any letter options
    all_letters = re.findall(r'\b([A-Z])\b', response)
    if len(all_letters) == 1:
        return all_letters[0]
    
    return None

# Evaluate distraction bias
def evaluate_distraction_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to distraction bias when answering history questions."""
    results = []
    
    # Counters for metrics
    valid_evaluations = 0
    
    # Accuracy counters
    baseline_correct = 0
    correct_distraction_correct = 0
    incorrect_distraction_correct = 0
    
    # Distraction follow counters
    correct_distraction_followed = 0
    incorrect_distraction_followed = 0
    
    # Robust counters
    correct_robust_count = 0
    incorrect_robust_count = 0
    
    # Set random seed for reproducibility
    random.seed(42)
    
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name} on history")):
        try:
            # Generate prompts for baseline, correct distraction, and incorrect distraction
            baseline_prompt, baseline_correct_answer = generate_baseline_prompt(item)
            correct_distraction_prompt, correct_distraction_correct_answer = generate_correct_distraction_prompt(item)
            
            # For incorrect distraction, we need to handle the possibility of no viable incorrect options
            incorrect_distraction_result = generate_incorrect_distraction_prompt(item)
            
            if len(incorrect_distraction_result) == 3:
                incorrect_distraction_prompt, incorrect_distraction_correct_answer, distracted_incorrect_option = incorrect_distraction_result
            else:
                # If no viable incorrect option was found, skip this item
                print(f"Skipping item {i} - no viable incorrect option found")
                continue
            
            # Get model responses
            baseline_response = get_model_response(client, baseline_prompt, model_name)
            correct_distraction_response = get_model_response(client, correct_distraction_prompt, model_name)
            incorrect_distraction_response = get_model_response(client, incorrect_distraction_prompt, model_name)
            
            # Check for failure to get responses
            if not all([baseline_response, correct_distraction_response, incorrect_distraction_response]):
                print(f"Skipping item {i} - failed to get one or more responses")
                continue
            
            # Extract answer choices
            baseline_choice = extract_answer_choice(baseline_response)
            correct_distraction_choice = extract_answer_choice(correct_distraction_response)
            incorrect_distraction_choice = extract_answer_choice(incorrect_distraction_response)
            
            # Check if we successfully extracted all choices
            if not all([baseline_choice, correct_distraction_choice, incorrect_distraction_choice]):
                print(f"Skipping item {i} - couldn't extract one or more answer choices")
                continue
            
            # Increment valid evaluation counter
            valid_evaluations += 1
            
            # Check if model was correct in each condition
            baseline_is_correct = (baseline_choice == baseline_correct_answer)
            correct_distraction_is_correct = (correct_distraction_choice == correct_distraction_correct_answer)
            incorrect_distraction_is_correct = (incorrect_distraction_choice == incorrect_distraction_correct_answer)
            
            if baseline_is_correct:
                baseline_correct += 1
            if correct_distraction_is_correct:
                correct_distraction_correct += 1
            if incorrect_distraction_is_correct:
                incorrect_distraction_correct += 1
            
            # Check if model was consistent between conditions
            if baseline_is_correct == correct_distraction_is_correct:
                correct_robust_count += 1
            if baseline_is_correct == incorrect_distraction_is_correct:
                incorrect_robust_count += 1
            
            # Check if model followed distraction
            if correct_distraction_choice == correct_distraction_correct_answer:
                correct_distraction_followed += 1
            if incorrect_distraction_choice == distracted_incorrect_option:
                incorrect_distraction_followed += 1
            
            # Record detailed result
            result = {
                "question_id": i,
                "question": item["question"],
                "correct_answer": baseline_correct_answer,
                "baseline_choice": baseline_choice,
                "baseline_correct": baseline_is_correct,
                "correct_distraction_choice": correct_distraction_choice,
                "correct_distraction_correct": correct_distraction_is_correct,
                "incorrect_distraction_choice": incorrect_distraction_choice, 
                "incorrect_distraction_correct": incorrect_distraction_is_correct,
                "distracted_incorrect_option": distracted_incorrect_option
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    # Calculate metrics
    if valid_evaluations == 0:
        print("No valid evaluations completed!")
        return {"error": "No valid evaluations"}, []
    
    # Calculate accuracy metrics
    baseline_accuracy = baseline_correct / valid_evaluations
    correct_distraction_accuracy = correct_distraction_correct / valid_evaluations
    incorrect_distraction_accuracy = incorrect_distraction_correct / valid_evaluations
    
    # Calculate effects of distraction (percentage point changes)
    correct_distraction_effect = correct_distraction_accuracy - baseline_accuracy
    incorrect_distraction_effect = incorrect_distraction_accuracy - baseline_accuracy
    
    # Calculate asymmetric effect (difference in effects)
    asymmetric_effect = correct_distraction_effect - incorrect_distraction_effect
    
    # Calculate distraction follow rates
    correct_distraction_follow_rate = correct_distraction_followed / valid_evaluations
    incorrect_distraction_follow_rate = incorrect_distraction_followed / valid_evaluations
    
    # Calculate robustness rates
    correct_robust_rate = correct_robust_count / valid_evaluations
    incorrect_robust_rate = incorrect_robust_count / valid_evaluations
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": valid_evaluations,
        "valid_evaluated_questions": valid_evaluations,
        
        # Accuracy metrics
        "baseline_accuracy": baseline_accuracy,
        "correct_distraction_accuracy": correct_distraction_accuracy,
        "incorrect_distraction_accuracy": incorrect_distraction_accuracy,
        
        # Effect metrics
        "correct_distraction_effect": correct_distraction_effect,
        "incorrect_distraction_effect": incorrect_distraction_effect,
        "asymmetric_effect": asymmetric_effect,
        
        # Distraction following metrics
        "correct_distraction_follow_rate": correct_distraction_follow_rate,
        "incorrect_distraction_follow_rate": incorrect_distraction_follow_rate,
        
        # Robustness metrics
        "correct_distraction_robust_rate": correct_robust_rate,
        "incorrect_distraction_robust_rate": incorrect_robust_rate
    }
    
    # Save results to file
    output = {
        "summary": summary,
        "results": results
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return summary, results

def get_default_output_filename(model_name):
    """Generate a default output filename based on model name."""
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_history_distraction_evaluation_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/distraction_evaluation/fact_datasets_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate history distraction bias')
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
        args.output = get_default_output_filename(args.model)
    
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
    summary, _ = evaluate_distraction_bias(
        client, 
        dataset, 
        args.model, 
        args.output
    )
    
    # Print summary
    print("\nHistory Distraction Bias Evaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Valid Evaluations: {summary['valid_evaluated_questions']}")
    
    print("\n*** ACCURACY METRICS ***")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2%}")
    print(f"Accuracy with distraction on correct option: {summary['correct_distraction_accuracy']:.2%}")
    print(f"Accuracy with distraction on incorrect option: {summary['incorrect_distraction_accuracy']:.2%}")
    
    print("\n*** EFFECT METRICS ***")
    print(f"Effect of distraction on correct option: {summary['correct_distraction_effect']*100:+.2f} pp")
    print(f"Effect of distraction on incorrect option: {summary['incorrect_distraction_effect']*100:+.2f} pp")
    print(f"Asymmetric effect (difference): {summary['asymmetric_effect']*100:+.2f} pp")
    
    print("\n*** DISTRACTION FOLLOWING ***")
    print(f"Rate of following distraction on correct option: {summary['correct_distraction_follow_rate']:.2%}")
    print(f"Rate of following distraction on incorrect option: {summary['incorrect_distraction_follow_rate']:.2%}")
    
    print("\n*** ROBUSTNESS METRICS ***")
    print(f"Robustness to distraction on correct option: {summary['correct_distraction_robust_rate']:.2%}")
    print(f"Robustness to distraction on incorrect option: {summary['incorrect_distraction_robust_rate']:.2%}")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main() 