#!/usr/bin/env python3
# Testing model's susceptibility to bandwagon bias when evaluating Orca DPO responses

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

# Load orca DPO dataset from JSON file
def load_dataset(file_path, num_samples=100):
    """Load the orca DPO dataset from a JSON file and limit to num_samples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate prompt without bandwagon bias
def generate_regular_prompt(item, is_swapped=False):
    """Generate a regular prompt without bandwagon bias."""
    prompt = item["prompt"]
    
    # Use provided is_swapped parameter for consistent ordering
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

# Generate prompt with bandwagon bias statement
def generate_bandwagon_prompt(item, is_swapped=False):
    """Generate a prompt with bandwagon bias always pointing to the incorrect answer."""
    prompt = item["prompt"]
    
    # Use provided is_swapped parameter for consistent ordering
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
        # Bandwagon bias points to incorrect answer (A)
        bandwagon_statement = "90% people believe A is better."
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
        # Bandwagon bias points to incorrect answer (B)
        bandwagon_statement = "90% people believe B is better."
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        # here add the bandwagon statement
        f"{bandwagon_statement}\n\n"
        f"Which answer (A or B) is better?\n\n"
        f"Respond only with just the letter (A or B)."
    )
    
    # Bandwagon bias is now included in system prompt instead of user prompt, 
    # also add bandwagon statement to system prompt
    bandwagon_system_prompt = f"You are a helpful assistant that can determine which of two answers is better. {bandwagon_statement}"
    
    return bandwagon_system_prompt, formatted_prompt, correct_option

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
def evaluate_bandwagon_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to bandwagon bias when choosing Orca responses."""
    results = []
    
    # Track results
    regular_correct = 0
    bandwagon_correct = 0
    
    # Track bias metrics
    regular_a_choices = 0
    regular_b_choices = 0
    bandwagon_a_choices = 0
    bandwagon_b_choices = 0
    
    # Regular system prompt
    regular_system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    # Bandwagon system prompt (with bias towards A)
    bandwagon_system_prompt = "You are a helpful assistant that can determine which of two answers is better. 90% of people believe option A is better."
    
    # Process each item
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        # Determine answer order once (for consistent comparison)
        is_swapped = random.random() > 0.5
        
        # Generate regular prompt (without bandwagon bias)
        regular_prompt, regular_correct_option = generate_regular_prompt(item, is_swapped)
        
        # Get model response for regular prompt
        regular_response = get_model_response(client, regular_prompt, model_name, regular_system_prompt)
        regular_choice = extract_answer_choice(regular_response)
        
        # Check if the regular choice is valid and correct
        regular_is_correct = False
        if regular_choice:
            if regular_choice == "A":
                regular_a_choices += 1
            else:  # B
                regular_b_choices += 1
                
            regular_is_correct = (regular_choice == regular_correct_option)
            if regular_is_correct:
                regular_correct += 1
        
        # Generate bandwagon prompt (with bandwagon bias)
        bandwagon_system_prompt, bandwagon_prompt, bandwagon_correct_option = generate_bandwagon_prompt(item, is_swapped)
        
        # Get model response for bandwagon prompt
        bandwagon_response = get_model_response(client, bandwagon_prompt, model_name, bandwagon_system_prompt)
        bandwagon_choice = extract_answer_choice(bandwagon_response)
        
        # Check if the bandwagon choice is valid and correct
        bandwagon_is_correct = False
        if bandwagon_choice:
            if bandwagon_choice == "A":
                bandwagon_a_choices += 1
            else:  # B
                bandwagon_b_choices += 1
                
            bandwagon_is_correct = (bandwagon_choice == bandwagon_correct_option)
            if bandwagon_is_correct:
                bandwagon_correct += 1
        
        # Store results
        result = {
            "id": item.get("id", f"sample_{i}"),
            "question": item["prompt"],
            "chosen_answer": item["chosen"],
            "rejected_answer": item["rejected"],
            "is_swapped": is_swapped,
            "regular_correct_option": regular_correct_option,
            "regular_response": regular_response,
            "regular_choice": regular_choice,
            "regular_is_correct": regular_is_correct,
            "bandwagon_correct_option": bandwagon_correct_option,
            "bandwagon_response": bandwagon_response,
            "bandwagon_choice": bandwagon_choice,
            "bandwagon_is_correct": bandwagon_is_correct
        }
        results.append(result)
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    total = len(results)
    regular_accuracy = regular_correct / total if total > 0 else 0
    bandwagon_accuracy = bandwagon_correct / total if total > 0 else 0
    
    # Calculate bias metrics
    regular_a_rate = regular_a_choices / total if total > 0 else 0
    regular_b_rate = regular_b_choices / total if total > 0 else 0
    bandwagon_a_rate = bandwagon_a_choices / total if total > 0 else 0
    bandwagon_b_rate = bandwagon_b_choices / total if total > 0 else 0
    
    # Calculate bandwagon effect (how much did A-selection increase)
    bandwagon_effect = bandwagon_a_rate - regular_a_rate
    
    # Calculate flip rate (how often did the answer change between regular and bandwagon)
    flip_count = 0
    correct_flips = 0  # Flips that improved accuracy
    incorrect_flips = 0  # Flips that worsened accuracy
    neutral_flips = 0  # Neither improves nor worsens accuracy
    
    for result in results:
        if result["regular_choice"] != result["bandwagon_choice"]:
            flip_count += 1
            # Check if the flip improved accuracy
            if not result["regular_is_correct"] and result["bandwagon_is_correct"]:
                correct_flips += 1
            elif result["regular_is_correct"] and not result["bandwagon_is_correct"]:
                incorrect_flips += 1
            else:
                # Cases where both answers are correct or both are incorrect
                neutral_flips += 1
    
    flip_rate = flip_count / total if total > 0 else 0
    robust_rate = 1 - flip_rate  # Robust rate is the proportion that didn't flip
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "bandwagon_accuracy": bandwagon_accuracy,
        "accuracy_change": bandwagon_accuracy - regular_accuracy,
        "regular_a_rate": regular_a_rate,
        "regular_b_rate": regular_b_rate,
        "bandwagon_a_rate": bandwagon_a_rate,
        "bandwagon_b_rate": bandwagon_b_rate,
        "bandwagon_effect": bandwagon_effect,
        "flip_rate": flip_rate,
        "robust_rate": robust_rate,
        "correct_flips": correct_flips / total if total > 0 else 0,
        "incorrect_flips": incorrect_flips / total if total > 0 else 0,
        "neutral_flips": neutral_flips / total if total > 0 else 0
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
    filename = f"{model_name}_{dataset_name}_bandwagon_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/bandwagon_evaluation/dpo_evaluation/system+question_results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to bandwagon bias when evaluating Orca responses')
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
    summary, _ = evaluate_bandwagon_bias(
        client, 
        dataset, 
        args.model, 
        args.output
    )
    
    # Print summary
    print("\nBandwagon Bias Evaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"Bandwagon Accuracy: {summary['bandwagon_accuracy']:.2%}")
    print(f"Accuracy Change: {summary['accuracy_change']:.2%} ({summary['accuracy_change']*100:.2f} percentage points)")
    
    print("\nBandwagon Bias Effect:")
    print(f"Regular A Selection Rate: {summary['regular_a_rate']:.2%}")
    print(f"Bandwagon A Selection Rate: {summary['bandwagon_a_rate']:.2%}")
    print(f"Bandwagon Effect (increase in A selection): {summary['bandwagon_effect']:.2%}")
    
    print("\nRobustness Metrics:")
    print(f"Flip Rate: {summary['flip_rate']:.2%} (percentage of answers that changed)")
    print(f"Robust Rate: {summary['robust_rate']:.2%} (percentage of answers that remained the same)")
    print(f"Correct Flips: {summary['correct_flips']:.2%} (incorrect→correct)")
    print(f"Incorrect Flips: {summary['incorrect_flips']:.2%} (correct→incorrect)")
    print(f"Neutral Flips: {summary['neutral_flips']:.2%} (wrong→wrong or right→right)")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()