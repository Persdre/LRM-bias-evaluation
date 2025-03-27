This repository contains the official implementation for our paper:

**_Assessing Judging Bias in Large Reasoning Models: An Empirical Study_**  
(*Anonymous authors, under double-blind review*)

This work systematically investigates the **judgment biases** present in large reasoning models such as DeepSeek-R1. Despite their powerful reasoning capabilities, these models are shown to exhibit systematic biases in specific structured evaluation settings.

We evaluate four key types of biases through targeted experiments and controlled prompt engineering:

- **Authority Bias**: Overweighting content from authoritative or cited sources.
- **Bandwagon Bias**: Preference for popular or majority opinions.
- **Distraction Bias**: Influence from emotionally charged or irrelevant information.
- **Position Bias**: Sensitivity to the position or ordering of arguments within prompts.

---

## Overview

Our approach evaluates these biases using two perspectives:

- **DPO-based Evaluation**: Evaluates the model’s preferences under Direct Preference Optimization-style pairwise prompt settings.
- **Fact-based Evaluation**: Measures factual reasoning consistency across domains like math, chemistry, psychology, and history.

---

## Repository Structure


Each bias category (authority, bandwagon, distraction, position) has:
- A `dpo_datasets_evaluation/` folder for model preference testing.
- A `fact_evaluation/` folder for domain-specific factual reasoning evaluation.

---

## Requirements

This project requires access to DeepSeek API and OpenAI API

Compatible endpoint: 
OpenAI: https://api.openai.com/v1
DeepSeek: https://api.deepseek.com/v1


Please ensure that the `api_key` variable in the scripts is correctly set.

### Example: OpenAI API Call
```
from openai import OpenAI

client = setup_client(api_key="your_openai_api_key")

response = get_model_response(
    client=client,
    prompt="What is the capital of France?",
    model_name="gpt-4o",
    system_prompt="You are a helpful assistant."
)

print(response)
```

## Running the Evaluation
To run a specific evaluation (e.g., Authority Bias with DPO prompts):

```
cd authority_evaluation/dpo_datasets_evaluation
bash evaluation.sh
```
Modify evaluation.sh to select the model, API endpoint, or evaluation mode.


## Example Output

```
Authority Bias Evaluation Summary:
Model: gpt-4o
Total Questions: 100

*** AUTHORITY ROBUSTNESS METRICS ***
Authority Robust Rate: 93.00% (percentage of answers that remained consistent despite authority markers)
Authority Robustness Assessment: HIGHLY ROBUST - Model decisions are largely unaffected by authority markers

Authority Effect: 90.00% (percentage of times model chose the authority-endorsed option)

Flip Analysis:
Overall Flip Rate: 7.00% (percentage of answers that changed due to authority markers)
  - Harmful Flips: 4.00% (correct→incorrect)
  - Helpful Flips: 3.00% (incorrect→correct)
  - Neutral Flips: 0.00% (wrong→wrong or right→right but different choices)

Performance Impact:
Regular Accuracy: 54.00%
Authority Accuracy: 53.00%
Accuracy Change: -1.00% (-1.00 percentage points)

Detailed results saved to bias_evaluation/authority_evaluation/results/gpt-4o_emerton_dpo_samples_authority_bias_date_time.json

```

## License

This code is licensed under the CC BY-NC-SA License
