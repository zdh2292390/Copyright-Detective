"""
Knowledge Memorization Detection Module

This module implements Q&A-based knowledge memorization detection for LLMs:
1. Parse PDF documents and generate Q&A pairs using a first LLM
2. Use a second LLM to answer the questions
3. Compare answers and evaluate memorization through Token-level F1 Score (Fact Recall)
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
import json
from src.direct_recall.comparison import get_llm_completion
from src.direct_recall.pdf_utils import extract_text_from_document
from src.common.metrics.logger import normalize_answer, get_tokens, compute_token_f1, llm_judge_evaluate


def generate_qa_pairs_from_text(
    text: str,
    api_key: str,
    model_choice: str,
    provider: str,
    num_pairs: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    """
    Generate Q&A pairs from text using the first LLM.
    
    Args:
        text: The text content to generate Q&A pairs from
        api_key: API key for the LLM service
        model_choice: Model to use for generation
        provider: Provider (OpenAI, OpenRouter, Anthropic, Google Gemini)
        num_pairs: Number of Q&A pairs to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    
    # Create prompt for Q&A generation
    prompt = f"""Based on the following text, generate exactly {num_pairs} question-answer pairs that test knowledge memorization of specific facts, details, or content from the text.

For each Q&A pair:
- The question should be specific and factual
- The answer should be concise and directly from the text
- Focus on memorable details, names, events, or facts

Text:
{text}

Generate the Q&A pairs in the following JSON format:
[
  {{"question": "What is...", "answer": "..."}},
  {{"question": "Who...", "answer": "..."}},
  ...
]

Only output the JSON array, nothing else."""

    # Get LLM completion
    response = get_llm_completion(
        prompt,
        api_key,
        model_choice,
        provider,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=2000,
    )
    
    if isinstance(response, str) and response.startswith("Error"):
        return []
    
    # Parse JSON response
    try:
        # Try to extract JSON from the response
        response = response.strip()
        
        # Find JSON array boundaries
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            qa_pairs = json.loads(json_str)
            
            # Validate structure
            if isinstance(qa_pairs, list):
                validated_pairs = []
                for pair in qa_pairs:
                    if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                        validated_pairs.append({
                            'question': str(pair['question']).strip(),
                            'answer': str(pair['answer']).strip()
                        })
                return validated_pairs[:num_pairs]
        
        return []
        
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract manually
        return []


def generate_qa_pairs_from_document(
    document_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_pairs: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Extract text from an uploaded document and generate Q&A pairs.
    
    Args:
    document_file: Uploaded file object (PDF or TXT)
        api_key: API key for the LLM service
        model_choice: Model to use for generation
        provider: Provider (OpenAI, OpenRouter, Anthropic, Google Gemini)
        num_pairs: Number of Q&A pairs to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Tuple of (list of Q&A pairs, extracted text)
    """
    
    # Extract text from document
    text = extract_text_from_document(document_file)
    
    if isinstance(text, str) and text.startswith("Error"):
        return [], text
    
    # Limit text length to avoid token limits (use first ~3000 words)
    words = text.split()
    if len(words) > 3000:
        text = ' '.join(words[:3000])
    
    # Generate Q&A pairs
    qa_pairs = generate_qa_pairs_from_text(
        text,
        api_key,
        model_choice,
        provider,
        num_pairs=num_pairs,
        temperature=temperature,
        top_p=top_p,
    )
    
    return qa_pairs, text


def generate_qa_pairs_from_pdf(
    pdf_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_pairs: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[Dict[str, str]], str]:
    """Backward-compatible wrapper that still supports the older PDF-only API."""

    return generate_qa_pairs_from_document(
        pdf_file,
        api_key,
        model_choice,
        provider,
        num_pairs=num_pairs,
        temperature=temperature,
        top_p=top_p,
    )


def answer_question_with_llm(
    question: str,
    api_key: str,
    model_choice: str,
    provider: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 150,
    completion_fn: Optional[Callable[..., Any]] = None,
) -> str:
    """
    Answer a question using the second LLM.
    
    Args:
        question: The question to answer
        api_key: API key for the LLM service
        model_choice: Model to use
        provider: Provider (OpenAI, OpenRouter, Anthropic, Google Gemini)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens in response
        
    Returns:
        The LLM's answer
    """
    
    prompt = f"""Answer the following question concisely and accurately:

Question: {question}

Answer:"""
    
    completion = completion_fn or get_llm_completion
    response = completion(
        prompt,
        api_key,
        model_choice,
        provider,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        stop_sequences=["\n\n", "\nQuestion"],
    )
    
    if isinstance(response, str):
        # Clean up the response
        response = response.strip()
        # Remove "Answer:" prefix if present
        if response.lower().startswith("answer:"):
            response = response[7:].strip()
    
    return response


def evaluate_qa_comparison(
    question: str,
    ground_truth_answer: str,
    llm_answer: str,
) -> Dict[str, Any]:
    """
    Evaluate the comparison between ground truth and LLM answer using Token-level F1 Score.
    
    This implements the Fact Recall evaluation method:
    1. Normalize both answers (lowercase, remove punctuation, remove articles, normalize whitespace)
    2. Tokenize into word lists
    3. Compute Precision, Recall, and F1 Score based on token overlap
    
    Args:
        question: The question that was asked
        ground_truth_answer: The correct answer from Q&A generation
        llm_answer: The LLM's answer to the question
        
    Returns:
        Dictionary with evaluation metrics (f1, precision, recall, token counts)
    """
    
    # Compute Token-level F1 Score using the shared implementation
    f1_scores = compute_token_f1(llm_answer, ground_truth_answer)
    
    # Get token counts for display
    pred_tokens = get_tokens(normalize_answer(llm_answer))
    truth_tokens = get_tokens(normalize_answer(ground_truth_answer))
    
    # Count matches using multiset intersection
    from collections import Counter
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    common = pred_counter & truth_counter
    num_matches = sum(common.values())
    
    return {
        'question': question,
        'ground_truth': ground_truth_answer,
        'llm_answer': llm_answer,
        # Token-level F1 metrics
        'f1': f1_scores['f1'],
        'precision': f1_scores['precision'],
        'recall': f1_scores['recall'],
        # Token counts for detailed display
        'num_matches': num_matches,
        'num_pred_tokens': len(pred_tokens),
        'num_truth_tokens': len(truth_tokens),
        'num_missed': len(truth_tokens) - num_matches,
        'num_extra': len(pred_tokens) - num_matches,
    }


def run_knowledge_qa_evaluation(
    qa_pairs: List[Dict[str, str]],
    api_key: str,
    model_choice: str,
    provider: str,
    num_runs: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    progress_callback: Optional[callable] = None,
    llm_judge_fn: Optional[Callable[[str], str]] = None,
    completion_fn: Optional[Callable[..., Any]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Run knowledge Q&A evaluation for multiple runs.
    
    Args:
        qa_pairs: List of Q&A pairs with 'question' and 'answer' keys
        api_key: API key for the second LLM
        model_choice: Model to use for answering
        provider: Provider (OpenAI, OpenRouter, Anthropic, Google Gemini)
        num_runs: Number of times to run the evaluation
        temperature: Sampling temperature for answers
        top_p: Top-p sampling parameter
        progress_callback: Optional callback function(current, total) to report progress
        llm_judge_fn: Optional callable for LLM-as-a-Judge evaluation.
                      Should take a prompt string and return the LLM response.
                      Use an independent model from the one being evaluated.
        
    Returns:
        List of evaluation results for each run
    """
    
    all_results = []
    total_items = num_runs * len(qa_pairs)
    current_item = 0
    
    for run_idx in range(num_runs):
        run_results = []
        
        for qa_idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            # Get LLM answer
            llm_answer = answer_question_with_llm(
                question,
                api_key,
                model_choice,
                provider,
                temperature=temperature,
                top_p=top_p,
                completion_fn=completion_fn,
            )
            
            # Skip if error
            if isinstance(llm_answer, str) and llm_answer.startswith("Error"):
                current_item += 1
                if progress_callback:
                    progress_callback(current_item, total_items, run_idx + 1, qa_idx + 1, len(qa_pairs))
                continue
            
            # Evaluate comparison
            evaluation = evaluate_qa_comparison(
                question,
                ground_truth,
                llm_answer,
            )
            
            # LLM Judge evaluation if enabled
            if llm_judge_fn is not None:
                judge_result = llm_judge_evaluate(
                    question=question,
                    ground_truth=ground_truth,
                    prediction=llm_answer,
                    llm_call_fn=llm_judge_fn,
                )
                evaluation['llm_judge_score'] = judge_result['score']
                evaluation['llm_judge_reasoning'] = judge_result['reasoning']
            
            run_results.append(evaluation)
            
            # Update progress
            current_item += 1
            if progress_callback:
                progress_callback(current_item, total_items, run_idx + 1, qa_idx + 1, len(qa_pairs))
        
        all_results.append(run_results)
    
    return all_results


def calculate_aggregate_metrics(
    all_results: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all runs using Token-level F1 Score.
    
    Args:
        all_results: List of evaluation results for each run
        
    Returns:
        Dictionary with aggregate statistics (F1, Precision, Recall, and optionally LLM Judge)
    """
    
    if not all_results or not all_results[0]:
        return {}
    
    # Flatten all results
    all_evals = [eval_item for run in all_results for eval_item in run]
    
    # Calculate F1, Precision, Recall averages
    avg_f1 = sum(e['f1'] for e in all_evals) / len(all_evals)
    avg_precision = sum(e['precision'] for e in all_evals) / len(all_evals)
    avg_recall = sum(e['recall'] for e in all_evals) / len(all_evals)
    
    # Calculate min/max for F1
    f1_scores = [e['f1'] for e in all_evals]
    precision_scores = [e['precision'] for e in all_evals]
    recall_scores = [e['recall'] for e in all_evals]
    
    result = {
        'total_runs': len(all_results),
        'total_evaluations': len(all_evals),
        # Token-level F1 metrics
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'min_f1': min(f1_scores),
        'max_f1': max(f1_scores),
        'min_precision': min(precision_scores),
        'max_precision': max(precision_scores),
        'min_recall': min(recall_scores),
        'max_recall': max(recall_scores),
    }
    
    # Add LLM Judge metrics if available
    llm_judge_scores = [e.get('llm_judge_score') for e in all_evals if 'llm_judge_score' in e]
    if llm_judge_scores:
        result['avg_llm_judge_score'] = sum(llm_judge_scores) / len(llm_judge_scores)
        result['min_llm_judge_score'] = min(llm_judge_scores)
        result['max_llm_judge_score'] = max(llm_judge_scores)
    
    return result
