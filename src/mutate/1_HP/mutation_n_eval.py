import pandas as pd
import time
import os
import requests
import json
from openai import OpenAI
import openai 
import replicate
import csv
from rouge_score import rouge_scorer
from tqdm import tqdm
import logging

# Initialize OpenAI, Anthropic, and Replicate clients
openai_client = OpenAI(
    # TODO: Add your OpenAI API key here
    # api_key="your_openai_api_key_here",
    # TODO: Add your base URL here if using a different endpoint
    # base_url="your_base_url_here"
)

# Load the persuasion template from the JSON file
with open('./outputs/1_persuasion_technique_template/correct_persuasion_framework_final.json', 'r') as file:
    persuasion_template = json.load(file)

# Define output CSV file for technique progress
# Configuration section
technique_progress = './mutate/1_HP/technique_progress.csv'

# Function to save evaluation data incrementally to a CSV file (Checkpointing)
def save_incremental_results(evaluation_data, checkpoint_file):
    df = pd.DataFrame(evaluation_data)
    if not os.path.exists(checkpoint_file):
        df.to_csv(checkpoint_file, index=False, mode='w')  # Create new file with header
    else:
        df.to_csv(checkpoint_file, index=False, mode='a', header=False)  # Append to existing file

# Main function to handle mutation and filtering process for all strategies
def generate_mutations_with_intention_check_all_strategies(original_questions):
    results = []
# Configuration section
    '''
    save_file = './outputs/2_persuasion_prompts/1_HP/Ethos/6_mutate_with_gpt4o_judgements_best_strategies_extract_HP_FT.csv'

    # Initialize CSV file
    save_to_csv(save_file, [], append=False)  # Create and write the header
    '''
    new_progress = [{
            'technique': strategy,
            'question1': 0,
            'question2': 0,
            'question3': 0,
            'question4': 0,
            'question5': 0,
            'question6': 0
        }]
    question_progress = 0
    if not os.path.exists(technique_progress):
        new_progress.to_csv(technique_progress, index=False, mode='w')  # Create new file with header
    else:
        progress = pd.read_csv(technique_progress)
        if strategy in progress['technique'].values:
            strategy_row = progress[progress['technique'] == strategy]
            zero_count = (strategy_row.iloc[0, 1:] == 0).sum()
            if zero_count == 0:
                pass
                # continue
            else:
                question_progress = 5 - zero_count
    # Iterate over each strategy in the persuasion template
    for strategy in persuasion_template.keys():
# Configuration section
        if strategy == "Ethos" or strategy == "Foot-in-the-Door": # deletion
            continue # deletion

        print(f"Processing strategy: {strategy}")   
        for original_text in original_questions:
            valid_mutations = []
            total_mutation_attempts = 0
            
            while len(valid_mutations) < 100:
                mutated_versions = persuasion_demo_mutation(original_text, strategy)
                total_mutation_attempts += len(mutated_versions)
                
                if total_mutation_attempts > 100:
                    print(f"Cannot get valid mutation for {strategy} on original text '{original_text}'. Skipping this strategy.")
                    break
                
                batch_results = []  
                
                for mutated_text in mutated_versions:
                    if len(valid_mutations) >= 50:
                        break
                    
                    # judgement = judge_core_intention(original_text, mutated_text, "gpt")
                    judgement = judge_core_intention(original_text, mutated_text, "gpt")
                    #judgement = judge_core_intention(original_text, mutated_text, "gpt")
                    batch_results.append([strategy, original_text, mutated_text, judgement])
                    
                    if judgement == "yes":
                        valid_mutations.append(mutated_text)
                
                save_to_csv(save_file, batch_results, append=True)
    
    return results
