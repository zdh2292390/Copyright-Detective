import pandas as pd
from oversample_labels_fn import generate_permutations
import sys
import os
from tqdm import tqdm
import glob

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_accuracy(row, row_name_1, row_name_2):
    return 1 if row[row_name_1] == row[row_name_2] else 0


def process_files(data_type, passage_size):

    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    

    if data_type == "BookTection":
        data_dir = os.path.join(script_dir, f'DECOP_{data_type}_{passage_size}')
    else:
        data_dir = os.path.join(script_dir, f'DECOP_{data_type}')

    if not os.path.exists(data_dir):
        print(f"There are no results for {data_type}. Run '2_decop_blackbox' first.")
        sys.exit()

    # Pattern to match files containing 'Paraphrases_Oversampling' in their name
    if data_type == "BookTection":
        pattern = f"{data_dir}/*Paraphrases_Oversampling_{passage_size}.xlsx"
    else:
        pattern = f"{data_dir}/*Paraphrases_Oversampling*.xlsx"

    print(pattern)
    books = []
    overall_accuracy_ChatGPT = []
    overall_accuracy_Claude2_1 = []
    overall_accuracy_LLaMA2_7B = []
    overall_accuracy_LLaMA2_13B = []
    overall_accuracy_LLaMA2_70B = []
    Label = []

    # Iterate over all matching Excel files in the folder
    for excel_file in glob.glob(pattern):
        # Load the Excel file into a pandas DataFrame
        data = pd.read_excel(excel_file)
        
        # Extract just the file name from the path
        document_name = os.path.basename(excel_file)
        
        # Print the name of the current Excel file (without the path)
        print(f"Loaded file: {document_name}")


        df = pd.DataFrame(data)

        if 'Max_Label_NoDebias' in df.columns:
            df['Accuracy_ChatGPT'] = df.apply(lambda row: calculate_accuracy(row, "True Answer", "Max_Label_NoDebias"), axis=1)
            accuracy_ChatGPT = df['Accuracy_ChatGPT'].mean()
            overall_accuracy_ChatGPT.append(accuracy_ChatGPT)
        else:
            overall_accuracy_ChatGPT.append(None)  # Append None if column not present

        if 'Claude2.1' in df.columns:
            df['Accuracy_Claude2.1'] = df.apply(lambda row: calculate_accuracy(row, "True Answer", "Claude2.1"), axis=1)
            accuracy_Claude2_1 = df['Accuracy_Claude2.1'].mean()
            overall_accuracy_Claude2_1.append(accuracy_Claude2_1)
        else:
            overall_accuracy_Claude2_1.append(None)  # Append None if column not present

        if 'Max_Label_NoDebias_LLMMA2-7B' in df.columns:
            df['Accuracy_LLaMA2-7B'] = df.apply(lambda row: calculate_accuracy(row, "True Answer", "Max_Label_NoDebias_LLAMA2-7B"), axis=1)
            accuracy_LLaMA2_7B = df['Accuracy_LLaMA2-7B'].mean()
            overall_accuracy_LLaMA2_7B.append(accuracy_LLaMA2_7B)
        else:
            overall_accuracy_LLaMA2_7B.append(None)  # Append None if column not present

        if 'Max_Label_NoDebias_LLAMA2-13B' in df.columns:
            df['Accuracy_LLaMA2-13B'] = df.apply(lambda row: calculate_accuracy(row, "True Answer", "Max_Label_NoDebias_LLAMA2-13B"), axis=1)
            accuracy_LLaMA2_13B = df['Accuracy_LLaMA2-13B'].mean()
            overall_accuracy_LLaMA2_13B.append(accuracy_LLaMA2_13B)
        else:
            overall_accuracy_LLaMA2_13B.append(None)  # Append None if column not present

        if 'Max_Label_NoDebias_LLAMA2-70B' in df.columns:
            df['Accuracy_LLaMA2-70B'] = df.apply(lambda row: calculate_accuracy(row, "True Answer", "Max_Label_NoDebias_LLAMA2-70B"), axis=1)
            accuracy_LLaMA2_70B = df['Accuracy_LLaMA2-70B'].mean()
            overall_accuracy_LLaMA2_70B.append(accuracy_LLaMA2_70B)
        else:
            overall_accuracy_LLaMA2_70B.append(None)  # Append None if column not present

        books.append(document_name)
        Label.append(data.loc[0, 'Label'])

    # Create a DataFrame
    final_results = {
        "Book Name": books,
        "Chat-GPT": overall_accuracy_ChatGPT,
        "Claude_2.1": overall_accuracy_Claude2_1,
        "LLaMA2-7B": overall_accuracy_LLaMA2_7B,
        "LLaMA2-13B": overall_accuracy_LLaMA2_13B,
        "LLaMA2-70B": overall_accuracy_LLaMA2_70B,
        "Label": Label
    }

    final_results = pd.DataFrame(final_results)
    final_results = final_results.sort_values(by='Label')  

    # Check if some columns are all filled with None
    columns_to_drop = [col for col in final_results.columns if all(pd.isnull(final_results[col]))]

    # Drop columns that are all None
    final_results = final_results.drop(columns=columns_to_drop)


    if (data_type == "BookTection"):
        final_results.to_excel(f"4_results_{data_type}_{passage_size}.xlsx", index=False)
    else:
        final_results.to_excel(f"4_results_{data_type}.xlsx", index=False)


    columns_to_iterate = [
        'Chat-GPT',
        'Claude_2.1',
        'LLaMA2-7B',
        'LLaMA2-13B',
        'LLaMA2-70B']

    colors = sns.color_palette("colorblind", n_colors=len(columns_to_iterate))
    plt.figure(figsize=(10, 8))
   
    for idx, col in enumerate(columns_to_iterate):
        if col in final_results.columns:
            new_columns = ["True_Label", "Predicted_Label"]
            auxiliar_df = final_results[['Label', col]]
            auxiliar_df.columns = new_columns

            fpr, tpr, thresholds = roc_curve(auxiliar_df['True_Label'], auxiliar_df['Predicted_Label'])
            roc_auc = auc(fpr, tpr)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            # Generating binary predictions
            binary_predictions = np.where(auxiliar_df['Predicted_Label'] >= optimal_threshold, 1, 0)
            # Calculating AUC for the binary predictions
            roc_auc = roc_auc_score(auxiliar_df['True_Label'], binary_predictions)

            # Plotting the ROC Curve with a unique color for each curve
            print(f'{col} - ROC curve (area = {roc_auc:.4f})')
            plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{col} - ROC curve (area = {roc_auc:.4f})')

    plt.title("ROC Curves")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python <name_of_file.py> --data <data_file> [--length <passage_size>]")
        print("<passage_size> is only mandatory for BookTection and should be one of: <small>, <medium>, or <large>")
        sys.exit(1)

    data_index = sys.argv.index("--data")    
    data_type = sys.argv[data_index + 1]



    if data_type == "BookTection":
        if "--length" not in sys.argv:
            print("Passage size (--length) is mandatory for BookTection data.")
            sys.exit(1)
        passage_size_index = sys.argv.index("--length")
        passage_size = sys.argv[passage_size_index + 1]

        if passage_size not in ["small", "medium", "large"]:
            print("Invalid passage_size. Available options are: <small>, <medium>, or <large>")
            sys.exit(1)
    elif data_type == "arXivTection":
        # For arXivTection data, set passage_size to a default value
        passage_size = "default_value"  # Replace with an appropriate default value
    else:
        print("Invalid data_file. Available options are: BookTection or arXivTection")
        sys.exit(1)

    process_files(data_type, passage_size)
