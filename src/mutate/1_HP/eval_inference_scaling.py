import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Adjust the path and others.')
parser.add_argument('--book', type=str, help='select the book')
parser.add_argument('--technique', type=str, help='select the technique')
args = parser.parse_args()

eva_result_dir = f"""./outputs/3_evaluation_results/{args.book}/{args.technique}"""

'''

python data_statistics.py --book "book" --technique "technique"
python data_statistics.py --book "1_HP" --technique "16_Foot-in-the-Door" --original_rouge_score 0.189021 0.102437 0.245976 0.140608

'''

inference_scaling_subname = "inference_scaling"
columns_of_rouge = ['rouge1_precision', 'rougeL_precision']
inference_scaling_list_for_claude = []

for dirpath, dirnames, filenames in os.walk(eva_result_dir):
    for filename in filenames:
        if inference_scaling_subname in filename:
            inference_scaling_path = os.path.join(eva_result_dir, filename)
            inference_scaling_df = pd.read_csv(inference_scaling_path)
            reduced_inference_scaling_df_for_claude = inference_scaling_df [columns_of_rouge]
            inference_scaling_list_for_claude.append(reduced_inference_scaling_df_for_claude)

inference_scaling_combined_df_for_claude = pd.concat(inference_scaling_list_for_claude, ignore_index=True)

def plot_boxplots_with_distribution(df, model_name, col1="rouge1_precision", col2="rougeL_precision", threshold1=None, threshold2=None):
    """
    Computes distribution statistics for two columns in a DataFrame and visualizes them as two separate boxplots.
    Adds a horizontal threshold line to each boxplot.

    Parameters:
    - df: pandas DataFrame containing the data
    - col1: str, first column to analyze (e.g., 'rouge1_precision')
    - col2: str, second column to analyze (e.g., 'rougeL_precision')
    - threshold1: float (optional), threshold for col1 (rouge1_precision)
    - threshold2: float (optional), threshold for col2 (rougeL_precision)

    Returns:
    - None (displays the plot)
    """

    # 1. Ensure the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Column {col1} or {col2} not found in DataFrame. Please check column names.")

    # 2. Extract data, dropping NaN values
    data1 = df[col1].dropna().values  
    data2 = df[col2].dropna().values  

    # 3. Compute distribution statistics
    stats = df[[col1, col2]].describe(percentiles=[0.25, 0.5, 0.75])

    stats_dict = {
        col1: {
            'min': stats.loc['min', col1],
            'q1': stats.loc['25%', col1],
            'median': stats.loc['50%', col1],
            'q3': stats.loc['75%', col1],
            'max': stats.loc['max', col1],
            'mean': stats.loc['mean', col1]
        },
        col2: {
            'min': stats.loc['min', col2],
            'q1': stats.loc['25%', col2],
            'median': stats.loc['50%', col2],
            'q3': stats.loc['75%', col2],
            'max': stats.loc['max', col2],
            'mean': stats.loc['mean', col2]
        }
    }

    # 4. Create two separate boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Two side-by-side plots

    # --- Boxplot for col1 (rouge1_precision) ---
    ax = axes[0]
    ax.boxplot(df[col1].dropna(), vert=True, patch_artist=True)
    ax.set_title(f"{col1} Distribution of {model_name}", fontsize=14)
    ax.set_ylabel("Value")

    # Add mean annotation
    mean_value = stats_dict[col1]['mean']
    ax.scatter(1, mean_value, color='red', zorder=3, label=f"Mean = {mean_value:.3f}")  
    ax.text(1.1, mean_value, f"Mean: {mean_value:.3f}", color='red', fontsize=10)

    # Add threshold line for rouge1_precision
    if threshold1 is not None:
        ax.axhline(y=threshold1, color='blue', linestyle='--', linewidth=2, label=f"Threshold = {threshold1}")

    ax.legend(loc="upper right", fontsize=10)

    # --- Boxplot for col2 (rougeL_precision) ---
    ax = axes[1]
    ax.boxplot(df[col2].dropna(), vert=True, patch_artist=True)
    ax.set_title(f"{col2} Distribution of {model_name}", fontsize=14)
    ax.set_ylabel("Value")

    # Add mean annotation
    mean_value = stats_dict[col2]['mean']
    ax.scatter(1, mean_value, color='red', zorder=3, label=f"Mean = {mean_value:.3f}")  
    ax.text(1.1, mean_value, f"Mean: {mean_value:.3f}", color='red', fontsize=10)

    # Add threshold line for rougeL_precision
    if threshold2 is not None:
        ax.axhline(y=threshold2, color='green', linestyle='--', linewidth=2, label=f"Threshold = {threshold2}")

    ax.legend(loc="upper right", fontsize=10)

    # 7. Show the plot
    plt.tight_layout()
    plt.show()


plot_boxplots_with_distribution(inference_scaling_combined_df_for_claude, "claude-3-haiku-20240307_zero-shot", col1="rouge1_precision", col2="rougeL_precision", threshold1=None, threshold2=None)
# import pdb; pdb.set_trace()
