"""
DECOP (DEtection of COPyrighted content) Analysis Module

This module provides wrapper functions for running DECOP multiple-choice tests
to detect if an LLM has been trained on specific copyrighted materials.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm

# Add the decop directory to the path
REPO_ROOT = Path(__file__).resolve().parents[2]
DECOP_DIR = REPO_ROOT / "src" / "direct_recall" / "decop"
sys.path.insert(0, str(DECOP_DIR))

from oversample_labels_fn import generate_permutations


softmax = nn.Softmax(dim=0)
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def get_available_datasets() -> List[str]:
    """Get list of available DECOP datasets."""
    data_dir = DECOP_DIR / "data"
    datasets = []
    
    if (data_dir / "BookTection.csv").exists():
        datasets.append("BookTection")
    if (data_dir / "arXivTection.csv").exists():
        datasets.append("arXivTection")
    
    return datasets


def get_passage_sizes(data_type: str) -> List[str]:
    """Get available passage sizes for a dataset."""
    if data_type == "BookTection":
        return ["small", "medium", "large"]
    else:
        return []


def query_llm_chatgpt(
    query_data: pd.Series,
    document_name: str,
    author_name: str,
    data_type: str,
    client: OpenAI
) -> torch.Tensor:
    """Query ChatGPT for multiple-choice answer."""
    
    if data_type == "BookTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the "{document_name}" book by {author_name}?\nOptions:\n"""
    elif data_type == "arXivTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the arXiv paper "{document_name}"?\nOptions:\n"""
    else:
        extra_prompt = "Question: Which of the following passages is the original?\nOptions:\n"
    
    prompt = (
        extra_prompt + 
        'A. ' + str(query_data['Example_A']) + '\n' + 
        'B. ' + str(query_data['Example_B']) + '\n' + 
        'C. ' + str(query_data['Example_C']) + '\n' + 
        'D. ' + str(query_data['Example_D']) + '\n' + 
        'Answer: '
    )
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        seed=2319,
        logprobs=4,
        logit_bias={32: +100, 33: +100, 34: +100, 35: +100}
    )
    
    dict_probs = response.choices[0].logprobs.top_logprobs[0]
    logits = torch.tensor([
        dict_probs.get("A", -100),
        dict_probs.get("B", -100),
        dict_probs.get("C", -100),
        dict_probs.get("D", -100)
    ], dtype=torch.float32)
    probabilities = softmax(logits)
    
    return probabilities


def query_llm_claude(
    query_data: pd.Series,
    document_name: str,
    author_name: str,
    data_type: str,
    anthropic_client: Anthropic
) -> str:
    """Query Claude for multiple-choice answer."""
    
    qa_prompt = """You are entering a multiple choice questions exam. You should directly answer each question by choosing the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding to the correct answer."""
    
    if data_type == "BookTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the "{document_name}" book by {author_name}?\nOptions:\n"""
    elif data_type == "arXivTection":
        extra_prompt = f"""Question: Which of the following passages is verbatim from the arXiv paper "{document_name}"?\nOptions:\n"""
    else:
        extra_prompt = "Question: Which of the following passages is the original?\nOptions:\n"
    
    prompt = (
        qa_prompt + extra_prompt + 
        'A. ' + str(query_data['Example_A']) + '\n' + 
        'B. ' + str(query_data['Example_B']) + '\n' + 
        'C. ' + str(query_data['Example_C']) + '\n' + 
        'D. ' + str(query_data['Example_D'])
    )
    
    completion = anthropic_client.completions.create(
        model="claude-2",
        max_tokens_to_sample=1,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT} Answer: ",
        temperature=0
    )
    
    return completion.completion.strip()


def run_decop_evaluation(
    data_type: str,
    model_name: str,
    api_key: str,
    passage_size: Optional[str] = None,
    progress_callback=None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Run DECOP evaluation on the selected dataset.
    
    Args:
        data_type: "BookTection" or "arXivTection"
        model_name: "ChatGPT" or "Claude"
        api_key: API key for the selected model
        passage_size: Required for BookTection ("small", "medium", or "large")
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Tuple of (success: bool, message: str, output_dir: Optional[Path])
    """
    
    # Validate inputs
    if data_type not in ["BookTection", "arXivTection"]:
        return False, "Invalid data type. Choose BookTection or arXivTection.", None
    
    if model_name not in ["ChatGPT", "Claude"]:
        return False, "Invalid model. Choose ChatGPT or Claude.", None
    
    if data_type == "BookTection" and not passage_size:
        return False, "Passage size is required for BookTection.", None
    
    if data_type == "BookTection" and passage_size not in ["small", "medium", "large"]:
        return False, "Invalid passage size. Choose small, medium, or large.", None
    
    # Initialize API client
    try:
        if model_name == "ChatGPT":
            client = OpenAI(api_key=api_key)
        else:
            anthropic_client = Anthropic(api_key=api_key)
    except Exception as e:
        return False, f"Failed to initialize API client: {str(e)}", None
    
    # Load dataset
    data_path = DECOP_DIR / "data" / f"{data_type}.csv"
    if not data_path.exists():
        return False, f"Dataset file not found: {data_path}", None
    
    try:
        document = pd.read_csv(data_path)
    except Exception as e:
        return False, f"Failed to load dataset: {str(e)}", None
    
    # Filter by passage size for BookTection
    if data_type == "BookTection":
        document = document[document['Length'] == passage_size]
        document = document.reset_index(drop=True)
    
    # Get unique document IDs
    unique_ids = document['ID'].unique().tolist()
    
    # Create output directory
    if data_type == "BookTection":
        out_dir = DECOP_DIR / f'DECOP_{data_type}_{passage_size}'
    else:
        out_dir = DECOP_DIR / f'DECOP_{data_type}'
    
    out_dir.mkdir(exist_ok=True)
    
    # Process each document
    total_docs = len(unique_ids)
    for i, document_id in enumerate(unique_ids):
        if progress_callback:
            progress_callback(i / total_docs, f"Processing document {i+1}/{total_docs}: {document_id}")
        
        # Prepare output file
        if data_type == "BookTection":
            file_out = out_dir / f'{document_id}_Paraphrases_Oversampling_{passage_size}.xlsx'
        else:
            file_out = out_dir / f'{document_id}_Paraphrases_Oversampling.xlsx'
        
        # Check if already processed
        if file_out.exists():
            document_aux = pd.read_excel(file_out)
        else:
            document_aux = document[document['ID'] == document_id]
            document_aux = document_aux.reset_index(drop=True)
            document_aux = generate_permutations(document_df=document_aux)
        
        # Extract document name and author
        if data_type == "BookTection":
            parts = document_id.split('_-_')
            doc_name = parts[0].replace('_', ' ')
            author_name = parts[1].replace('_', ' ') if len(parts) > 1 else ""
        else:
            doc_name = document_id
            author_name = ""
        
        # Query LLM for each question
        if model_name == "ChatGPT":
            A_probs, B_probs, C_probs, D_probs, max_labels = [], [], [], [], []
            
            for j in range(len(document_aux)):
                probabilities = query_llm_chatgpt(
                    document_aux.iloc[j],
                    doc_name,
                    author_name,
                    data_type,
                    client
                )
                A_probs.append(probabilities[0].item())
                B_probs.append(probabilities[1].item())
                C_probs.append(probabilities[2].item())
                D_probs.append(probabilities[3].item())
                max_labels.append(mapping.get(torch.argmax(probabilities).item(), 'Unknown'))
            
            document_aux["A_Probability"] = A_probs
            document_aux["B_Probability"] = B_probs
            document_aux["C_Probability"] = C_probs
            document_aux["D_Probability"] = D_probs
            document_aux["Max_Label_NoDebias"] = max_labels
        else:
            max_labels = []
            for j in range(len(document_aux)):
                answer = query_llm_claude(
                    document_aux.iloc[j],
                    doc_name,
                    author_name,
                    data_type,
                    anthropic_client
                )
                max_labels.append(answer)
            
            document_aux["Claude2.1"] = max_labels
        
        # Save results
        document_aux.to_excel(file_out, index=False)
    
    if progress_callback:
        progress_callback(1.0, f"Completed! Processed {total_docs} documents.")
    
    return True, f"Successfully processed {total_docs} documents. Results saved to {out_dir}", out_dir


def calculate_results(
    data_type: str,
    passage_size: Optional[str] = None
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Calculate accuracy and ROC metrics from DECOP results.
    
    Args:
        data_type: "BookTection" or "arXivTection"
        passage_size: Required for BookTection
    
    Returns:
        Tuple of (success: bool, message: str, results_df: Optional[pd.DataFrame])
    """
    
    # Determine results directory
    if data_type == "BookTection":
        if not passage_size:
            return False, "Passage size required for BookTection", None
        results_dir = DECOP_DIR / f'DECOP_{data_type}_{passage_size}'
        pattern = f"*Paraphrases_Oversampling_{passage_size}.xlsx"
    else:
        results_dir = DECOP_DIR / f'DECOP_{data_type}'
        pattern = "*Paraphrases_Oversampling*.xlsx"
    
    if not results_dir.exists():
        return False, f"Results directory not found: {results_dir}. Run evaluation first.", None
    
    # Find all result files
    import glob
    files = list(glob.glob(str(results_dir / pattern)))
    
    if not files:
        return False, f"No result files found in {results_dir}", None
    
    # Calculate accuracies
    books = []
    overall_accuracy_chatgpt = []
    overall_accuracy_claude = []
    labels = []
    
    def calculate_accuracy(row, col1, col2):
        return 1 if row[col1] == row[col2] else 0
    
    for excel_file in files:
        data = pd.read_excel(excel_file)
        df = pd.DataFrame(data)
        
        file_name = os.path.basename(excel_file)
        
        # ChatGPT accuracy
        if 'Max_Label_NoDebias' in df.columns and 'True Answer' in df.columns:
            df['Accuracy_ChatGPT'] = df.apply(
                lambda row: calculate_accuracy(row, "True Answer", "Max_Label_NoDebias"),
                axis=1
            )
            accuracy_chatgpt = df['Accuracy_ChatGPT'].mean()
            overall_accuracy_chatgpt.append(accuracy_chatgpt)
        else:
            overall_accuracy_chatgpt.append(None)
        
        # Claude accuracy
        if 'Claude2.1' in df.columns and 'True Answer' in df.columns:
            df['Accuracy_Claude2.1'] = df.apply(
                lambda row: calculate_accuracy(row, "True Answer", "Claude2.1"),
                axis=1
            )
            accuracy_claude = df['Accuracy_Claude2.1'].mean()
            overall_accuracy_claude.append(accuracy_claude)
        else:
            overall_accuracy_claude.append(None)
        
        books.append(file_name)
        labels.append(data.loc[0, 'Label'] if 'Label' in data.columns else None)
    
    # Create results DataFrame
    final_results = pd.DataFrame({
        "Document": books,
        "ChatGPT_Accuracy": overall_accuracy_chatgpt,
        "Claude_Accuracy": overall_accuracy_claude,
        "Label": labels
    })
    
    # Remove columns that are all None
    final_results = final_results.dropna(axis=1, how='all')
    
    # Sort by label if available
    if 'Label' in final_results.columns:
        final_results = final_results.sort_values(by='Label')
    
    return True, f"Successfully calculated results for {len(files)} documents", final_results
