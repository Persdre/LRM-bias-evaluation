#!/usr/bin/env python3
# Evaluate distraction bias by comparing effect of adding irrelevant information
# to correct options versus incorrect options for chemistry questions

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
    """Load the chemistry dataset from a JSON file and limit to num_samples if specified."""
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
    options[correct_index] = f"{options[correct_index]} (This researcher enjoys hiking on weekends.)"
    
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
    
    # Keep track of which incorrect option got the distraction
    distracted_option = option_letters[incorrect_index]
    
    return formatted_prompt, correct_answer, distracted_option

# Get model response with retry logic
def get_model_response(client, prompt, model_name, temperature=0.7, max_tokens=1024):
    """Get response from the model with retry logic."""
    system_prompt = "You are a helpful assistant that can answer chemistry questions accurately."
    
    max_retries = 3
    for attempt in range(max_retries):
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
            print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed after maximum retry attempts")
                return None

# Extract the answer choice (A, B, C, etc.) from the model's response
def extract_answer_choice(response):
    """Extract the answer choice (A, B, C, etc.) from the model's response."""
    if not response:
        return None
    
    # Check for different patterns to find the answer
    patterns = [
        r"(?:answer is|choice is|select|choose|option is|answer:|choice:|option:)\s*([A-Z])",
        r"(?:answer:|i choose|i select|final answer:|therefore:|thus:|conclusion:|in conclusion:).*?([A-Z])(?:\.|$)",
        r"\b([A-Z])\b"  # Look for a standalone A, B, C, etc.
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # If no pattern match, look for the first letter answer in the response
    for char in response:
        if char.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return char.upper()
    
    return None

def evaluate_distraction_bias(client, dataset, model_name, output_file):
    """Evaluate how distractions on correct vs. incorrect options affect model choices."""
    results = []
    
    # Tracking counters
    baseline_correct = 0
    correct_distraction_correct = 0
    incorrect_distraction_correct = 0
    
    # Tracking distraction following
    correct_distraction_followed = 0  # When model chooses the correct option with distraction
    incorrect_distraction_followed = 0  # When model chooses the incorrect option with distraction
    
    # Tracking robustness
    correct_robust_count = 0  # When baseline and correct distraction give same answer
    incorrect_robust_count = 0  # When baseline and incorrect distraction give same answer
    
    valid_evaluations = 0
    
    # Process each dataset item
    for item in tqdm(dataset, desc=f"Evaluating {model_name} on chemistry questions"):
        try:
            # 1. Get baseline response (no distraction)
            baseline_prompt, correct_answer = generate_baseline_prompt(item)
            baseline_response = get_model_response(client, baseline_prompt, model_name)
            baseline_choice = extract_answer_choice(baseline_response)
            
            if not baseline_choice:
                print(f"Warning: Could not extract choice from baseline response, skipping item")
                continue
                
            baseline_is_correct = (baseline_choice == correct_answer)
            if baseline_is_correct:
                baseline_correct += 1
            
            # 2. Get response with distraction on correct option
            correct_prompt, _ = generate_correct_distraction_prompt(item)
            correct_response = get_model_response(client, correct_prompt, model_name)
            correct_choice = extract_answer_choice(correct_response)
            
            if not correct_choice:
                print(f"Warning: Could not extract choice from correct distraction response, skipping item")
                continue
                
            correct_is_correct = (correct_choice == correct_answer)
            if correct_is_correct:
                correct_distraction_correct += 1
                correct_distraction_followed += 1  # Followed distraction AND got it right
            
            # Check if decision is robust (unchanged from baseline)
            if baseline_choice == correct_choice:
                correct_robust_count += 1
            
            # 3. Get response with distraction on incorrect option
            incorrect_prompt, _, distracted_option = generate_incorrect_distraction_prompt(item)
            incorrect_response = get_model_response(client, incorrect_prompt, model_name)
            incorrect_choice = extract_answer_choice(incorrect_response)
            
            if not incorrect_choice:
                print(f"Warning: Could not extract choice from incorrect distraction response, skipping item")
                continue
                
            incorrect_is_correct = (incorrect_choice == correct_answer)
            if incorrect_is_correct:
                incorrect_distraction_correct += 1
            
            # Check if the distraction was "followed" - did they choose the distracted (incorrect) option?
            if incorrect_choice == distracted_option:
                incorrect_distraction_followed += 1
            
            # Check if decision is robust (unchanged from baseline)
            if baseline_choice == incorrect_choice:
                incorrect_robust_count += 1
            
            # Store all results for this item
            result = {
                "question": item["question"],
                "options": item["options"],
                "correct_answer": correct_answer,
                "distracted_incorrect_option": distracted_option,
                
                # Baseline
                "baseline_prompt": baseline_prompt,
                "baseline_response": baseline_response,
                "baseline_choice": baseline_choice,
                "baseline_is_correct": baseline_is_correct,
                
                # Correct option distraction
                "correct_distraction_prompt": correct_prompt,
                "correct_distraction_response": correct_response,
                "correct_distraction_choice": correct_choice,
                "correct_distraction_is_correct": correct_is_correct,
                "correct_distraction_followed": (correct_choice == correct_answer and correct_is_correct),
                
                # Incorrect option distraction
                "incorrect_distraction_prompt": incorrect_prompt,
                "incorrect_distraction_response": incorrect_response,
                "incorrect_distraction_choice": incorrect_choice,
                "incorrect_distraction_is_correct": incorrect_is_correct,
                "incorrect_distraction_followed": (incorrect_choice == distracted_option)
            }
            
            results.append(result)
            valid_evaluations += 1
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
            
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    if valid_evaluations == 0:
        print("No valid evaluations completed!")
        return {"error": "No valid evaluations"}, []
    
    # Accuracy rates
    baseline_accuracy = baseline_correct / valid_evaluations
    correct_distraction_accuracy = correct_distraction_correct / valid_evaluations
    incorrect_distraction_accuracy = incorrect_distraction_correct / valid_evaluations
    
    # Effects on accuracy (percentage point changes)
    correct_distraction_effect = correct_distraction_accuracy - baseline_accuracy
    incorrect_distraction_effect = incorrect_distraction_accuracy - baseline_accuracy
    
    # Asymmetric effects
    asymmetric_effect = correct_distraction_effect - incorrect_distraction_effect
    
    # Distraction following rates
    correct_distraction_follow_rate = correct_distraction_followed / valid_evaluations
    incorrect_distraction_follow_rate = incorrect_distraction_followed / valid_evaluations
    
    # Robustness rates (consistency of answers)
    correct_robust_rate = correct_robust_count / valid_evaluations
    incorrect_robust_rate = incorrect_robust_count / valid_evaluations
    
    # Create summary
    summary = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(dataset),
        "valid_evaluated_questions": valid_evaluations,
        
        # Accuracy metrics
        "baseline_accuracy": baseline_accuracy,
        "correct_distraction_accuracy": correct_distraction_accuracy,
        "incorrect_distraction_accuracy": incorrect_distraction_accuracy,
        
        # Effect metrics
        "correct_distraction_effect": correct_distraction_effect,
        "incorrect_distraction_effect": incorrect_distraction_effect,
        "asymmetric_effect": asymmetric_effect,
        
        # Distraction following
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
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    return summary, results

def get_default_output_filename(model_name):
    """Generate a default output filename based on model name."""
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_chemistry_distraction_evaluation_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/distraction_evaluation/fact_datasets_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate chemistry distraction bias')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/mmlu_pro_samples/chemistry_samples.json',
                        help='Path to chemistry dataset JSON')
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
    print("\nChemistry Distraction Bias Evaluation Summary:")
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