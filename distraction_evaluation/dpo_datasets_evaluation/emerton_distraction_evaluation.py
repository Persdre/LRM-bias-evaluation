# Evaluate distraction bias by comparing effect of adding irrelevant information
# to correct options versus incorrect options
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

# Load dataset from JSON file
def load_dataset(file_path, num_samples=100):
    """Load the dataset from a JSON file and limit to num_samples."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    elif isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate baseline prompt (no distraction)
def generate_baseline_prompt(item, is_swapped=False):
    """Generate a baseline prompt without distraction."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    if is_swapped:
        first_answer = item["rejected"]  # Answer A
        second_answer = item["chosen"]   # Answer B
        correct_option = "B"
        incorrect_option = "A"
    else:
        first_answer = item["chosen"]    # Answer A
        second_answer = item["rejected"] # Answer B
        correct_option = "A"
        incorrect_option = "B"
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer}\n\n"
        f"B: {second_answer}\n\n"
        f"Which answer is better?\n\n"
        f"only respond with the letter (A or B)"
    )
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

# Generate prompt with distraction added to correct option
def generate_correct_distraction_prompt(item, is_swapped=False):
    """Add distraction statement to the correct option."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    if is_swapped:
        first_answer = item["rejected"]  # Answer A (incorrect)
        second_answer = item["chosen"]   # Answer B (correct)
        correct_option = "B"
        incorrect_option = "A"
        
        # Add distraction to the correct option (B)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"Answer B went hiking today.\n\n"
            f"Which answer is better?\n\n"
            f"only respond with the letter (A or B)"
        )
    else:
        first_answer = item["chosen"]    # Answer A (correct)
        second_answer = item["rejected"] # Answer B (incorrect)
        correct_option = "A"
        incorrect_option = "B"
        
        # Add distraction to the correct option (A)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"Which answer is better?\n\n"
            f"Answer A went hiking today.\n\n"
            f"only respond with the letter (A or B)"
        )
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

# Generate prompt with distraction added to incorrect option
def generate_incorrect_distraction_prompt(item, is_swapped=False):
    """Add distraction statement to the incorrect option."""
    prompt = item["input"]
    system_prompt = item["system"]
    
    if is_swapped:
        first_answer = item["rejected"]  # Answer A (incorrect)
        second_answer = item["chosen"]   # Answer B (correct)
        correct_option = "B"
        incorrect_option = "A"
        
        # Add distraction to the incorrect option (A)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"Answer A went hiking today.\n\n"
            f"Which answer is better?\n\n"
            f"only respond with the letter (A or B)"
        )
    else:
        first_answer = item["chosen"]    # Answer A (correct)
        second_answer = item["rejected"] # Answer B (incorrect)
        correct_option = "A"
        incorrect_option = "B"
        
        # Add distraction to the incorrect option (B)
        formatted_prompt = (
            f"Question: {prompt}\n\n"
            f"A: {first_answer}\n\n"
            f"B: {second_answer}\n\n"
            f"Answer B went hiking today.\n\n"
            f"Which answer is better?\n\n"
            f"only respond with the letter (A or B)"
        )
    
    return system_prompt, formatted_prompt, correct_option, incorrect_option

# Get model response with retry logic
def get_model_response(client, system_prompt, prompt, model_name, temperature=0.7, max_tokens=1024):
    """Get response from the model with retry logic."""
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

# Extract the answer choice (A or B) from the model's response
def extract_answer_choice(response):
    """Extract the answer choice (A or B) from the model's response."""
    if not response:
        return None
    
    # Check for different patterns to find the answer
    patterns = [
        r"(?:answer is|choice is|select|choose|option is|answer:|choice:|option:)\s*([AB])",
        r"(?:answer:|i choose|i select|final answer:|therefore:|thus:|conclusion:|in conclusion:).*?([AB])(?:\.|$)",
        r"\b([AB])\b"  # Look for a standalone A or B
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # If no pattern match, look for the first A or B in the response
    for char in response:
        if char.upper() in ["A", "B"]:
            return char.upper()
    
    return None

def evaluate_distraction_bias(client, dataset, model_name, output_file):
    """Evaluate how distractions on correct vs. incorrect options affect model choices."""
    results = []
    
    # Tracking counters
    baseline_correct = 0
    correct_distraction_correct = 0
    incorrect_distraction_correct = 0
    
    correct_distraction_followed = 0
    incorrect_distraction_followed = 0
    
    correct_robust_count = 0
    incorrect_robust_count = 0
    
    valid_evaluations = 0
    
    # Process each dataset item
    for item in tqdm(dataset, desc=f"Evaluating {model_name}"):
        try:
            # Randomize option order (A/B) to avoid position bias
            is_swapped = random.random() > 0.5
            
            # 1. Get baseline response (no distraction)
            baseline_system, baseline_prompt, correct_option, incorrect_option = generate_baseline_prompt(item, is_swapped)
            baseline_response = get_model_response(client, baseline_system, baseline_prompt, model_name)
            baseline_choice = extract_answer_choice(baseline_response)
            
            if not baseline_choice:
                print(f"Warning: Could not extract choice from baseline response, skipping item")
                continue
                
            baseline_is_correct = (baseline_choice == correct_option)
            if baseline_is_correct:
                baseline_correct += 1
            
            # 2. Get response with distraction on correct option
            correct_system, correct_prompt, _, _ = generate_correct_distraction_prompt(item, is_swapped)
            correct_response = get_model_response(client, correct_system, correct_prompt, model_name)
            correct_choice = extract_answer_choice(correct_response)
            
            if not correct_choice:
                print(f"Warning: Could not extract choice from correct distraction response, skipping item")
                continue
                
            correct_is_correct = (correct_choice == correct_option)
            if correct_is_correct:
                correct_distraction_correct += 1
                
            # Check if distraction was "followed" - did they choose the distracted option?
            if correct_choice == correct_option:  # If they chose the option with distraction
                correct_distraction_followed += 1
            
            # Check if decision is robust (unchanged from baseline)
            if baseline_choice == correct_choice:
                correct_robust_count += 1
            
            # 3. Get response with distraction on incorrect option
            incorrect_system, incorrect_prompt, _, _ = generate_incorrect_distraction_prompt(item, is_swapped)
            incorrect_response = get_model_response(client, incorrect_system, incorrect_prompt, model_name)
            incorrect_choice = extract_answer_choice(incorrect_response)
            
            if not incorrect_choice:
                print(f"Warning: Could not extract choice from incorrect distraction response, skipping item")
                continue
                
            incorrect_is_correct = (incorrect_choice == correct_option)
            if incorrect_is_correct:
                incorrect_distraction_correct += 1
                
            # Check if distraction was "followed" - did they choose the distracted option?
            if incorrect_choice == incorrect_option:  # If they chose the option with distraction
                incorrect_distraction_followed += 1
            
            # Check if decision is robust (unchanged from baseline)
            if baseline_choice == incorrect_choice:
                incorrect_robust_count += 1
            
            # Store all results for this item
            result = {
                "question": item["input"],
                "chosen_answer": item["chosen"],
                "rejected_answer": item["rejected"],
                "is_swapped": is_swapped,
                "correct_option": correct_option,
                "incorrect_option": incorrect_option,
                
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
                "correct_distraction_followed": (correct_choice == correct_option),
                
                # Incorrect option distraction
                "incorrect_distraction_prompt": incorrect_prompt,
                "incorrect_distraction_response": incorrect_response,
                "incorrect_distraction_choice": incorrect_choice,
                "incorrect_distraction_is_correct": incorrect_is_correct,
                "incorrect_distraction_followed": (incorrect_choice == incorrect_option)
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
    dataset_name = os.path.basename(dataset_path).replace('.json', '')
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{dataset_name}_distraction_results_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/distraction_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Compare effects of adding distractions to correct vs. incorrect options')
    parser.add_argument('--dataset', type=str, default='bias_evaluation/filtered_samples/emerton_dpo_samples.json',
                        help='Path to dataset JSON')
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
    
    # Set up client
    client = setup_client(args.api_key, args.api_base)
    
    # Load dataset with sample limit
    dataset = load_dataset(args.dataset, args.samples)
    print(f"Loaded dataset with {len(dataset)} questions")
    
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
    
    print("\nAccuracy Metrics:")
    print(f"Baseline Accuracy (No distraction): {summary['baseline_accuracy']:.2%}")
    print(f"Accuracy with Distraction on Correct Option: {summary['correct_distraction_accuracy']:.2%}")
    print(f"Accuracy with Distraction on Incorrect Option: {summary['incorrect_distraction_accuracy']:.2%}")
    
    print("\nDistraction Effect on Accuracy:")
    print(f"Effect when Distracting Correct Option: {summary['correct_distraction_effect']:.2%} points")
    print(f"Effect when Distracting Incorrect Option: {summary['incorrect_distraction_effect']:.2%} points")
    
    print("\nDistraction Follow Rates:")
    print(f"Rate of Following Distraction on Correct Option: {summary['correct_distraction_follow_rate']:.2%}")
    print(f"Rate of Following Distraction on Incorrect Option: {summary['incorrect_distraction_follow_rate']:.2%}")
    
    print("\nRobustness Metrics:")
    print(f"Robust Rate (Correct Distraction): {summary['correct_robust_rate']:.2%}")
    print(f"Robust Rate (Incorrect Distraction): {summary['incorrect_robust_rate']:.2%}")
    print(f"Overall Robust Rate: {summary['overall_robust_rate']:.2%}")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()