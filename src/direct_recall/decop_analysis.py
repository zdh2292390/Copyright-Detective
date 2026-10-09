"""
Dataset Analysis Module

This module provides wrapper functions for running single-choice multiple-choice tests
to detect if an LLM has been trained on specific copyrighted materials.
"""

import os
import math
import tempfile
import re
import sys
import pandas as pd
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm
from src.api_concurrency import limit_api_concurrency
from src.anthropic_utils import create_anthropic_message, extract_anthropic_response_text

# Add the data directory to the path
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "direct_recall" / "decop" / "data"
sys.path.insert(0, str(DATA_DIR.parent))

from oversample_labels_fn import generate_permutations


softmax = nn.Softmax(dim=0)
mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


def get_available_datasets() -> List[str]:
    """Get list of available datasets."""
    datasets = []
    
    if (DATA_DIR / "BookTection.csv").exists():
        datasets.append("BookTection")
    if (DATA_DIR / "arXivTection.csv").exists():
        datasets.append("arXivTection")
    
    return datasets


def get_passage_sizes(data_type: str) -> List[str]:
    """Get available passage sizes for a dataset."""
    if data_type == "BookTection":
        return ["small", "medium", "large"]
    else:
        return []


DECOP_OPENAI_MODEL = "gpt-4o-mini"
DECOP_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _parse_chat_top_logprobs(response) -> Dict[str, float]:
    """Extract A/B/C/D logprobs from a chat completion response."""
    dict_probs: Dict[str, float] = {}
    try:
        logprobs_content = response.choices[0].logprobs
        if not logprobs_content or not getattr(logprobs_content, "content", None):
            return dict_probs
        top_logprobs = logprobs_content.content[0].top_logprobs or []
        for item in top_logprobs:
            token = getattr(item, "token", None)
            logprob = getattr(item, "logprob", None)
            if token is not None and logprob is not None:
                value = float(logprob)
                label = str(token).strip().upper()
                if label in mapping.values() and math.isfinite(value) and value <= 0:
                    dict_probs[label] = value
    except Exception:
        pass
    return dict_probs


def _query_llm_chatgpt_uncached(
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
    
    with limit_api_concurrency(timeout=120):
        response = client.chat.completions.create(
            model=DECOP_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=4,
        )


    dict_probs = _parse_chat_top_logprobs(response)
    if not dict_probs:
        raise ValueError('Model returned no valid A/B/C/D log probabilities.')
    logits = torch.tensor([
        dict_probs.get("A", -100),
        dict_probs.get("B", -100),
        dict_probs.get("C", -100),
        dict_probs.get("D", -100)
    ], dtype=torch.float32)
    probabilities = softmax(logits)
    
    return probabilities


def _query_llm_claude_uncached(
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
    
    with limit_api_concurrency(timeout=120):
        response = create_anthropic_message(anthropic_client, {
            "model": DECOP_ANTHROPIC_MODEL,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        })

    answer = extract_anthropic_response_text(response)
    match = re.fullmatch(r"[A-D][.)]?", answer, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Model returned no valid single-choice answer.")
    return answer[0].upper()


def _decop_checkpoint(operation, payload, invoke, is_success):
    try:
        from src.resumable_analysis import checkpoint_call
    except ModuleNotFoundError as exc:
        if exc.name != "src.resumable_analysis":
            raise
        return invoke()
    return checkpoint_call(operation, payload, invoke, is_success=is_success)


def _decop_payload(query_data, document_name, author_name, data_type, model, provider):
    return {
        "options": {label: str(query_data[f"Example_{label}"]) for label in mapping.values()},
        "document_name": document_name, "author_name": author_name,
        "data_type": data_type, "model": model, "provider": provider,
        "temperature": 0, "max_tokens": 1,
    }


def query_llm_chatgpt(query_data: pd.Series, document_name: str, author_name: str,
                      data_type: str, client: OpenAI) -> torch.Tensor:
    """Save JSON probabilities and restore the original float32 tensor API."""
    payload = _decop_payload(query_data, document_name, author_name, data_type, DECOP_OPENAI_MODEL, "OpenAI")
    payload["endpoint"] = str(getattr(client, "base_url", "https://api.openai.com/v1"))
    values = _decop_checkpoint(
        "decop.openai", payload,
        lambda: _query_llm_chatgpt_uncached(query_data, document_name, author_name, data_type, client).tolist(),
        lambda result: isinstance(result, list) and len(result) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 <= value <= 1 for value in result)
            and sum(result) > 0,
    )
    return torch.tensor(values, dtype=torch.float32)


def query_llm_claude(query_data: pd.Series, document_name: str, author_name: str,
                     data_type: str, anthropic_client: Anthropic) -> str:
    payload = _decop_payload(query_data, document_name, author_name, data_type, DECOP_ANTHROPIC_MODEL, "Anthropic")
    payload["endpoint"] = str(getattr(anthropic_client, "base_url", "https://api.anthropic.com"))
    return _decop_checkpoint(
        "decop.anthropic", payload,
        lambda: _query_llm_claude_uncached(query_data, document_name, author_name, data_type, anthropic_client),
        lambda result: isinstance(result, str) and result in mapping.values(),
    )


def _raise_checkpoint_error(exc):
    try:
        from src.resumable_analysis import AnalysisCheckpointError
    except ModuleNotFoundError as missing:
        if missing.name != "src.resumable_analysis":
            raise
        return
    if isinstance(exc, AnalysisCheckpointError):
        raise exc


_DATASET_EVALUATION_LOCK = Lock()


def run_dataset_evaluation(
    data_type: str,
    model_name: str,
    api_key: str,
    passage_size: Optional[str] = None,
    progress_callback=None,
) -> Tuple[bool, str, Optional[Path]]:
    """Evaluate a dataset without overlapping writes to its result workbooks."""
    if not _DATASET_EVALUATION_LOCK.acquire(blocking=False):
        return False, "Another dataset evaluation is writing results. Retry after it finishes.", None
    try:
        return _run_dataset_evaluation(
            data_type, model_name, api_key, passage_size, progress_callback
        )
    except Exception as exc:
        _raise_checkpoint_error(exc)
        return False, f"Evaluation failed: {type(exc).__name__}: {exc}", None
    finally:
        _DATASET_EVALUATION_LOCK.release()


def _run_dataset_evaluation(
    data_type: str,
    model_name: str,
    api_key: str,
    passage_size: Optional[str] = None,
    progress_callback=None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Run evaluation on the selected dataset.
    
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
    
    # Load dataset
    data_path = DATA_DIR / f"{data_type}.csv"
    if not data_path.exists():
        return False, f"Dataset file not found: {data_path}", None
    
    try:
        document = pd.read_csv(data_path)
    except Exception as e:
        return False, f"Failed to load dataset: {str(e)}", None
    
    required = {'ID', 'Example_A', 'Example_B', 'Example_C', 'Example_D', 'Answer'}
    if not required.issubset(document.columns) or (data_type == "BookTection" and "Length" not in document.columns):
        return False, "Dataset is missing required columns.", None

    # Filter by passage size for BookTection
    if data_type == "BookTection":
        document = document[document['Length'] == passage_size]
        document = document.reset_index(drop=True)
    
    required = {'ID', 'Example_A', 'Example_B', 'Example_C', 'Example_D', 'Answer'}
    if not required.issubset(document.columns) or document.empty:
        return False, "Dataset has no matching questions or is missing required columns.", None
    # Get unique document IDs
    unique_ids = document['ID'].unique().tolist()
    
    # Create output directory
    if data_type == "BookTection":
        out_dir = DATA_DIR / f'results_{data_type}_{passage_size}'
    else:
        out_dir = DATA_DIR / f'results_{data_type}'
    
    out_dir.mkdir(exist_ok=True)
    
    # Initialize API client
    try:
        if model_name == "ChatGPT":
            client = OpenAI(api_key=api_key, timeout=120, max_retries=0)
        else:
            client = anthropic_client = Anthropic(api_key=api_key, timeout=120, max_retries=0)
    except Exception as e:
        return False, f"Failed to initialize API client: {str(e)}", None
    
    try:
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

            source = document[document['ID'] == document_id].reset_index(drop=True)
            expected = generate_permutations(document_df=source)
            document_aux = pd.read_excel(file_out) if file_out.exists() else expected
            source_columns = list(expected.columns)
            if (not set(source_columns).issubset(document_aux.columns)
                or not document_aux[source_columns].fillna("").astype(str).reset_index(drop=True).equals(
                    expected[source_columns].fillna("").astype(str).reset_index(drop=True))):
                document_aux = expected

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
            with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".xlsx", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                document_aux.to_excel(temporary_path, index=False)
                os.replace(temporary_path, file_out)
            finally:
                temporary_path.unlink(missing_ok=True)

        if progress_callback:
            progress_callback(1.0, f"Completed! Processed {total_docs} documents.")

        return True, f"Successfully processed {total_docs} documents. Results saved to {out_dir}", out_dir
    except Exception as exc:
        _raise_checkpoint_error(exc)
        return False, f"Evaluation failed: {type(exc).__name__}: {exc}", out_dir
    finally:
        try:
            client.close()
        except Exception:
            pass


def calculate_results(
    data_type: str,
    passage_size: Optional[str] = None
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Calculate accuracy and ROC metrics from evaluation results.
    
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
        results_dir = DATA_DIR / f'results_{data_type}_{passage_size}'
        pattern = f"*Paraphrases_Oversampling_{passage_size}.xlsx"
    else:
        results_dir = DATA_DIR / f'results_{data_type}'
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
