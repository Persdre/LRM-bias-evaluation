#!/usr/bin/env python3
# Testing model's susceptibility to distraction bias when evaluating truthfulness

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

# Load truthful DPO dataset from JSON file
def load_dataset(file_path, num_samples=100):
    """Load the truthful DPO dataset from a JSON file and limit to num_samples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate baseline prompt (no distraction)
def generate_baseline_prompt(item, is_swapped=False):
    """Generate a baseline prompt without distraction."""
    prompt = item["prompt"]
    
    # Use provided is_swapped parameter for consistent ordering
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
        incorrect_option = "A"
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
        incorrect_option = "B"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer (A or B) is better?\n\n"
        f"Respond only with just the letter (A or B)."
    )
    
    # System prompt
    system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

# Generate prompt with distraction added to correct option
def generate_correct_distraction_prompt(item, is_swapped=False):
    """Add distraction statement to the correct option."""
    prompt = item["prompt"]
    system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
        incorrect_option = "A"
        
        # Add distraction to the correct option (B)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"The person who wrote answer B went hiking today.\n\n"
            f"Which answer (A or B) is better?\n\n"
            f"Respond only with just the letter (A or B)."
        )
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
        incorrect_option = "B"
        
        # Add distraction to the correct option (A)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"The person who wrote answer A went hiking today.\n\n"
            f"Which answer (A or B) is better?\n\n"
            f"Respond only with just the letter (A or B)."
        )
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

# Generate prompt with distraction added to incorrect option
def generate_incorrect_distraction_prompt(item, is_swapped=False):
    """Add distraction statement to the incorrect option."""
    prompt = item["prompt"]
    system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
        incorrect_option = "A"
        
        # Add distraction to the incorrect option (A)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"The person who wrote answer A went hiking today.\n\n"
            f"Which answer (A or B) is better?\n\n"
            f"Respond only with just the letter (A or B)."
        )
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
        incorrect_option = "B"
        
        # Add distraction to the incorrect option (B)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"The person who wrote answer B went hiking today.\n\n"
            f"Which answer (A or B) is better?\n\n"
            f"Respond only with just the letter (A or B)."
        )
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

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
            stream=False,
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
def evaluate_distraction_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to distraction bias when choosing truthful answers."""
    results = []
    
    # Track results
    baseline_correct = 0
    correct_distraction_correct = 0
    incorrect_distraction_correct = 0
    
    # Track distraction effects
    correct_distraction_followed = 0
    incorrect_distraction_followed = 0
    
    # Track robust decisions
    correct_robust_count = 0
    incorrect_robust_count = 0
    
    # Set random seed for reproducibility in answer ordering
    random.seed(42)
    
    # Track valid evaluations
    valid_evaluations = 0
    
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        # Skip invalid items
        if not all(k in item for k in ["prompt", "chosen", "rejected"]):
            print(f"Skipping item {i} - missing required fields")
            continue
            
        # Determine position order once for each item (for consistency)
        is_swapped = random.random() > 0.5
        
        try:
            # Baseline approach (no distraction)
            system_prompt, baseline_prompt, baseline_correct_option, baseline_incorrect_option = generate_baseline_prompt(item, is_swapped)
            baseline_response = get_model_response(client, system_prompt, baseline_prompt, model_name)
            
            if not baseline_response:
                print(f"Skipping item {i} - failed to get baseline response")
                continue
                
            baseline_choice = extract_answer_choice(baseline_response)
            
            if not baseline_choice:
                print(f"Skipping item {i} - couldn't extract baseline choice")
                continue
                
            baseline_is_correct = baseline_choice == baseline_correct_option
            
            if baseline_is_correct:
                baseline_correct += 1
            
            # Distraction on correct option
            system_prompt, correct_prompt, correct_correct_option, correct_incorrect_option = generate_correct_distraction_prompt(item, is_swapped)
            correct_response = get_model_response(client, system_prompt, correct_prompt, model_name)
            
            if not correct_response:
                print(f"Skipping item {i} - failed to get correct distraction response")
                continue
                
            correct_choice = extract_answer_choice(correct_response)
            
            if not correct_choice:
                print(f"Skipping item {i} - couldn't extract correct distraction choice")
                continue
                
            correct_is_correct = correct_choice == correct_correct_option
            
            # Check if the model followed the distraction for correct option
            if correct_choice == correct_correct_option:
                correct_distraction_followed += 1
                
            if correct_is_correct:
                correct_distraction_correct += 1
                
            # Check if the model's decision is robust to distraction on correct option
            if (baseline_choice == correct_choice):
                correct_robust_count += 1
            
            # Distraction on incorrect option
            system_prompt, incorrect_prompt, incorrect_correct_option, incorrect_incorrect_option = generate_incorrect_distraction_prompt(item, is_swapped)
            incorrect_response = get_model_response(client, system_prompt, incorrect_prompt, model_name)
            
            if not incorrect_response:
                print(f"Skipping item {i} - failed to get incorrect distraction response")
                continue
                
            incorrect_choice = extract_answer_choice(incorrect_response)
            
            if not incorrect_choice:
                print(f"Skipping item {i} - couldn't extract incorrect distraction choice")
                continue
                
            incorrect_is_correct = incorrect_choice == incorrect_correct_option
            
            # Check if the model followed the distraction for incorrect option
            if incorrect_choice == incorrect_incorrect_option:
                incorrect_distraction_followed += 1
                
            if incorrect_is_correct:
                incorrect_distraction_correct += 1
                
            # Check if the model's decision is robust to distraction on incorrect option
            if (baseline_choice == incorrect_choice):
                incorrect_robust_count += 1
            
            # Store results
            result = {
                "id": item.get("id", f"sample_{i}"),
                "question": item["prompt"],
                "chosen_answer": item["chosen"],
                "rejected_answer": item["rejected"],
                "is_swapped": is_swapped,
                
                # Baseline results
                "baseline_prompt": baseline_prompt,
                "baseline_correct_option": baseline_correct_option,
                "baseline_response": baseline_response,
                "baseline_choice": baseline_choice,
                "baseline_is_correct": baseline_is_correct,
                
                # Correct distraction results
                "correct_distraction_prompt": correct_prompt,
                "correct_distraction_response": correct_response,
                "correct_distraction_choice": correct_choice,
                "correct_distraction_is_correct": correct_is_correct,
                "correct_distraction_followed": (correct_choice == correct_correct_option),
                
                # Incorrect distraction results
                "incorrect_distraction_prompt": incorrect_prompt,
                "incorrect_distraction_response": incorrect_response,
                "incorrect_distraction_choice": incorrect_choice,
                "incorrect_distraction_is_correct": incorrect_is_correct,
                "incorrect_distraction_followed": (incorrect_choice == incorrect_incorrect_option)
            }
            
            results.append(result)
            valid_evaluations += 1
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
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
    
    # Distraction follow rates
    correct_distraction_follow_rate = correct_distraction_followed / valid_evaluations
    incorrect_distraction_follow_rate = incorrect_distraction_followed / valid_evaluations
    
    # Robust rates (decisions unchanged by distraction)
    correct_robust_rate = correct_robust_count / valid_evaluations
    incorrect_robust_rate = incorrect_robust_count / valid_evaluations
    overall_robust_rate = (correct_robust_count + incorrect_robust_count) / (2 * valid_evaluations)
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": valid_evaluations,
        
        # Accuracy rates
        "baseline_accuracy": baseline_accuracy,
        "correct_distraction_accuracy": correct_distraction_accuracy,
        "incorrect_distraction_accuracy": incorrect_distraction_accuracy,
        
        # Effects on accuracy
        "correct_distraction_effect": correct_distraction_effect,
        "incorrect_distraction_effect": incorrect_distraction_effect,
        
        # Distraction follow rates
        "correct_distraction_follow_rate": correct_distraction_follow_rate,
        "incorrect_distraction_follow_rate": incorrect_distraction_follow_rate,
        
        # Robust rates
        "correct_robust_rate": correct_robust_rate,
        "incorrect_robust_rate": incorrect_robust_rate,
        "overall_robust_rate": overall_robust_rate
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
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_distraction_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/distraction_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s susceptibility to distraction bias when evaluating truthfulness')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/truthful_dpo_samples.json',
                        help='Path to truthful DPO dataset JSON')
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
    summary, _ = evaluate_distraction_bias(
        client, 
        dataset, 
        args.model, 
        args.output
    )
    
    # Print summary
    print("\nDistraction Bias Evaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    
    print("\n*** BASELINE PERFORMANCE ***")
    print(f"Baseline Accuracy: {summary['baseline_accuracy']:.2%}")
    
    print("\n*** DISTRACTION EFFECTS ***")
    print(f"When distraction added to CORRECT option:")
    print(f"  - Accuracy: {summary['correct_distraction_accuracy']:.2%} (change: {summary['correct_distraction_effect']*100:+.2f} pp)")
    print(f"  - Follow rate: {summary['correct_distraction_follow_rate']:.2%} (how often model chose the distracted correct option)")
    print(f"  - Robust rate: {summary['correct_robust_rate']:.2%} (decisions unchanged from baseline)")
    
    print(f"\nWhen distraction added to INCORRECT option:")
    print(f"  - Accuracy: {summary['incorrect_distraction_accuracy']:.2%} (change: {summary['incorrect_distraction_effect']*100:+.2f} pp)")
    print(f"  - Follow rate: {summary['incorrect_distraction_follow_rate']:.2%} (how often model chose the distracted incorrect option)")
    print(f"  - Robust rate: {summary['incorrect_robust_rate']:.2%} (decisions unchanged from baseline)")
    
    print(f"\nOverall robustness to distraction: {summary['overall_robust_rate']:.2%}")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main() 