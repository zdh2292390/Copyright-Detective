import pandas as pd
from itertools import permutations


def generate_permutations(document_df):
    """
    Generate all permutations of the answer options for each question.
    
    Args:
        document_df: DataFrame containing questions with columns:
            - Example_A, Example_B, Example_C, Example_D (the 4 options)
            - Answer (the correct answer letter: A, B, C, or D)
            - Other metadata columns
    
    Returns:
        DataFrame with permuted questions, where each original question
        generates 24 rows (4! = 24 permutations)
    """
    
    result_rows = []
    
    for idx, row in document_df.iterrows():
        # Get the four options
        options = {
            'A': row['Example_A'],
            'B': row['Example_B'],
            'C': row['Example_C'],
            'D': row['Example_D']
        }
        
        # Get the correct answer
        true_answer = row['Answer']
        
        # Generate all permutations of the option labels (A, B, C, D)
        labels = ['A', 'B', 'C', 'D']
        for perm in permutations(labels):
            # Create a new row with permuted options
            new_row = row.copy()
            
            # Map the permuted positions to the original options
            new_row['Example_A'] = options[perm[0]]
            new_row['Example_B'] = options[perm[1]]
            new_row['Example_C'] = options[perm[2]]
            new_row['Example_D'] = options[perm[3]]
            
            # Update the true answer to match the new position
            # Find where the original true answer ended up
            true_answer_new_position = labels[perm.index(true_answer)]
            new_row['True Answer'] = true_answer_new_position
            
            result_rows.append(new_row)
    
    return pd.DataFrame(result_rows).reset_index(drop=True)
