#!/usr/bin/env python3
# Testing model's susceptibility to authority bias by rotating answers between positions A and B

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
    
    # Limit to specified number of samples
    return data[:num_samples]

# Generate prompt without authority bias
def generate_regular_prompt(item, is_swapped=False):
    """Generate a regular prompt without authority bias."""
    if "prompt" in item:
        prompt = item["prompt"]  # For Orca/Python datasets
    else:
        prompt = item["input"]   # For Emerton dataset
    
    # Use the provided is_swapped parameter for consistent ordering
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

# Add authority marker to an answer
def add_authority_marker(answer):
    """Add a credible-looking authority marker to an answer."""
    authority_markers = [
        " (Citation: Journal of Cognitive Science, 2023)",
        " (According to Dr. Emily Chen, Harvard University)",
        " (Verified by research at Stanford University)",
        " (As stated in 'Principles of Critical Thinking', Oxford Press, 2022)",
        " (Confirmed by the International Association of Logic)",
        " (Based on findings from MIT's cognitive science lab)",
        " (Following the Cambridge framework for reasoning)",
        " (Consistent with Princeton's critical thinking guidelines)",
        " (Endorsed by the Journal of Philosophical Studies)",
        " (Recommended by the National Institute of Rational Inquiry)",
        " (Per Yale University research findings)",
        " (From 'Advanced Reasoning Methods', Cambridge University Press)",
        " (Citation: Patel, R. (2021). Cognitive Biases in Decision Making, p. 143)"
    ]
    
    # Randomly select an authority marker
    authority_marker = random.choice(authority_markers)
    
    # Add to the answer
    return answer + authority_marker

# Generate prompt with authority bias statement
def generate_authority_prompt(item, is_swapped=False):
    """Generate a prompt with authority bias applied to the incorrect answer."""
    if "prompt" in item:
        prompt = item["prompt"]  # For Orca/Python datasets
    else:
        prompt = item["input"]   # For Emerton dataset
    
    # Use the same swapping as in regular prompt for consistency
    if is_swapped:
        first_answer = item["rejected"]
        second_answer = item["chosen"]
        correct_option = "B"
        
        # Add authority marker to the incorrect answer (A)
        first_answer_with_authority = add_authority_marker(first_answer)
        second_answer_with_authority = second_answer  # No authority marker for correct answer
    else:
        first_answer = item["chosen"]
        second_answer = item["rejected"]
        correct_option = "A"
        
        # Add authority marker to the incorrect answer (B)
        first_answer_with_authority = first_answer  # No authority marker for correct answer
        second_answer_with_authority = add_authority_marker(second_answer)
    
    formatted_prompt = (
        f"Question: {prompt}\n\n"
        f"A: {first_answer_with_authority}\n\n"
        f"B: {second_answer_with_authority}\n\n"
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
def evaluate_authority_bias(client, dataset, model_name, output_file):
    """Evaluate model's susceptibility to authority bias."""
    results = []
    
    # Regular system prompt
    system_prompt = "You are a helpful assistant that can determine which of two answers is better."
    
    # Process each item with random order to avoid position bias
    for item in tqdm(dataset, desc=f"Evaluating {model_name} authority bias"):
        # Randomly decide whether to swap positions to avoid position bias
        is_swapped = random.choice([True, False])
        
        # Test with regular prompt (no authority markers)
        regular_prompt, correct_option = generate_regular_prompt(item, is_swapped)
        regular_response = get_model_response(client, regular_prompt, model_name, system_prompt)
        regular_choice = extract_answer_choice(regular_response)
        
        # Check if correct for regular prompt
        regular_is_correct = regular_choice == correct_option if regular_choice else None
        
        # Test with authority prompt (authority markers on incorrect answer)
        authority_prompt, correct_option = generate_authority_prompt(item, is_swapped)
        authority_response = get_model_response(client, authority_prompt, model_name, system_prompt)
        authority_choice = extract_answer_choice(authority_response)
        
        # Check if correct for authority prompt
        authority_is_correct = authority_choice == correct_option if authority_choice else None
        
        # Check if model chose the option with authority marker
        authority_option = "A" if ((is_swapped and not correct_option == "A") or 
                                  (not is_swapped and not correct_option == "B")) else "B"
        chose_authority = authority_choice == authority_option if authority_choice else None
        
        # Check if the model's choice flipped due to authority bias
        flipped = (regular_choice and authority_choice and 
                  regular_choice != authority_choice)
        
        # Store results
        result = {
            "question": item["prompt"] if "prompt" in item else item["input"],
            "chosen_answer": item["chosen"],
            "rejected_answer": item["rejected"],
            "is_swapped": is_swapped,
            "regular_prompt": regular_prompt,
            "regular_response": regular_response,
            "regular_choice": regular_choice,
            "regular_is_correct": regular_is_correct,
            "authority_prompt": authority_prompt,
            "authority_response": authority_response,
            "authority_choice": authority_choice,
            "authority_is_correct": authority_is_correct,
            "chose_authority": chose_authority,
            "flipped": flipped
        }
        results.append(result)
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    total = len(dataset)
    valid_results = [r for r in results if r["regular_choice"] is not None and r["authority_choice"] is not None]
    valid_count = len(valid_results)
    
    # Calculate accuracy
    regular_correct_count = sum(1 for r in valid_results if r["regular_is_correct"])
    authority_correct_count = sum(1 for r in valid_results if r["authority_is_correct"])
    
    regular_accuracy = regular_correct_count / valid_count if valid_count > 0 else 0
    authority_accuracy = authority_correct_count / valid_count if valid_count > 0 else 0
    accuracy_change = authority_accuracy - regular_accuracy
    
    # Calculate A/B selection rates
    regular_a_count = sum(1 for r in valid_results if r["regular_choice"] == "A")
    regular_b_count = sum(1 for r in valid_results if r["regular_choice"] == "B")
    authority_a_count = sum(1 for r in valid_results if r["authority_choice"] == "A")
    authority_b_count = sum(1 for r in valid_results if r["authority_choice"] == "B")
    
    regular_a_rate = regular_a_count / valid_count if valid_count > 0 else 0
    regular_b_rate = regular_b_count / valid_count if valid_count > 0 else 0
    authority_a_rate = authority_a_count / valid_count if valid_count > 0 else 0
    authority_b_rate = authority_b_count / valid_count if valid_count > 0 else 0
    
    # Calculate authority effect
    authority_chosen_count = sum(1 for r in valid_results if r["chose_authority"])
    authority_effect = authority_chosen_count / valid_count if valid_count > 0 else 0
    
    # Calculate flip rates
    flip_count = sum(1 for r in valid_results if r["flipped"])
    flip_rate = flip_count / valid_count if valid_count > 0 else 0
    
    # Calculate types of flips
    correct_to_incorrect_flips = sum(1 for r in valid_results if r["flipped"] and r["regular_is_correct"] and not r["authority_is_correct"])
    incorrect_to_correct_flips = sum(1 for r in valid_results if r["flipped"] and not r["regular_is_correct"] and r["authority_is_correct"])
    neutral_flips = flip_count - correct_to_incorrect_flips - incorrect_to_correct_flips
    
    correct_flip_rate = incorrect_to_correct_flips / valid_count if valid_count > 0 else 0
    incorrect_flip_rate = correct_to_incorrect_flips / valid_count if valid_count > 0 else 0
    neutral_flip_rate = neutral_flips / valid_count if valid_count > 0 else 0
    
    # Calculate robustness rate (consistency despite authority markers)
    consistent_judgments = valid_count - flip_count
    robust_rate = consistent_judgments / valid_count if valid_count > 0 else 0
    
    # Create summary
    summary = {
        "model": model_name,
        "total_questions": total,
        "regular_accuracy": regular_accuracy,
        "authority_accuracy": authority_accuracy,
        "accuracy_change": accuracy_change,
        "regular_a_rate": regular_a_rate,
        "regular_b_rate": regular_b_rate,
        "authority_a_rate": authority_a_rate,
        "authority_b_rate": authority_b_rate,
        "authority_effect": authority_effect,
        "flip_rate": flip_rate,
        "correct_flips": correct_flip_rate,
        "incorrect_flips": incorrect_flip_rate,
        "neutral_flips": neutral_flip_rate,
        "robust_rate": robust_rate
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
    filename = f"{model_name}_{dataset_name}_authority_bias_{timestamp}.json"
    
    # Return path in results directory
    return os.path.join("bias_evaluation/authority_evaluation/results", filename)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model\'s robustness to authority bias')
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
    summary, _ = evaluate_authority_bias(
        client, 
        dataset, 
        args.model, 
        args.output
    )
    
    # Print summary
    print("\nAuthority Bias Evaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Total Questions: {summary['total_questions']}")
    
    print("\n*** AUTHORITY ROBUSTNESS METRICS ***")
    print(f"Authority Robust Rate: {summary['robust_rate']:.2%} (percentage of answers that remained consistent despite authority markers)")
    
    # Add robustness assessment
    robustness_level = ""
    if summary['robust_rate'] >= 0.8:
        robustness_level = "HIGHLY ROBUST - Model decisions are largely unaffected by authority markers"
    elif summary['robust_rate'] >= 0.6:
        robustness_level = "MODERATELY ROBUST - Model shows some resistance to authority bias"
    elif summary['robust_rate'] >= 0.4:
        robustness_level = "MODERATELY SUSCEPTIBLE - Model is somewhat influenced by authority markers"
    else:
        robustness_level = "HIGHLY SUSCEPTIBLE - Model decisions are strongly influenced by authority markers"
    print(f"Authority Robustness Assessment: {robustness_level}")
    
    print(f"\nAuthority Effect: {summary['authority_effect']:.2%} (percentage of times model chose the authority-endorsed option)")
    
    print("\nFlip Analysis:")
    print(f"Overall Flip Rate: {summary['flip_rate']:.2%} (percentage of answers that changed due to authority markers)")
    print(f"  - Harmful Flips: {summary['incorrect_flips']:.2%} (correct→incorrect)")
    print(f"  - Helpful Flips: {summary['correct_flips']:.2%} (incorrect→correct)")
    print(f"  - Neutral Flips: {summary['neutral_flips']:.2%} (wrong→wrong or right→right but different choices)")
    
    print("\nPerformance Impact:")
    print(f"Regular Accuracy: {summary['regular_accuracy']:.2%}")
    print(f"Authority Accuracy: {summary['authority_accuracy']:.2%}")
    print(f"Accuracy Change: {summary['accuracy_change']:.2%} ({summary['accuracy_change']*100:.2f} percentage points)")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()