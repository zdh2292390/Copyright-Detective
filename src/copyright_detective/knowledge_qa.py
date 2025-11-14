"""
Knowledge Memorization Detection Module

This module implements Q&A-based knowledge memorization detection for LLMs:
1. Parse PDF documents and generate Q&A pairs using a first LLM
2. Use a second LLM to answer the questions
3. Compare answers and evaluate memorization through similarity metrics
"""

from typing import List, Dict, Tuple, Optional, Any
import json
from src.copyright_detective.comparison import (
    get_llm_completion,
    calculate_rouge_score,
    calculate_jaccard_index,
)
from src.copyright_detective.pdf_utils import extract_text_from_pdf
from Levenshtein import distance


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


def generate_qa_pairs_from_pdf(
    pdf_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_pairs: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Extract text from PDF and generate Q&A pairs.
    
    Args:
        pdf_file: Uploaded PDF file object
        api_key: API key for the LLM service
        model_choice: Model to use for generation
        provider: Provider (OpenAI, OpenRouter, Anthropic, Google Gemini)
        num_pairs: Number of Q&A pairs to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Tuple of (list of Q&A pairs, extracted text)
    """
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)
    
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


def answer_question_with_llm(
    question: str,
    api_key: str,
    model_choice: str,
    provider: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 150,
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
    
    response = get_llm_completion(
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
    Evaluate the comparison between ground truth and LLM answer.
    
    Args:
        question: The question that was asked
        ground_truth_answer: The correct answer from Q&A generation
        llm_answer: The LLM's answer to the question
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Truncate LLM answer to match ground truth length if it's longer
    if len(llm_answer) > len(ground_truth_answer):
        llm_answer = llm_answer[:len(ground_truth_answer)]
    
    # Calculate similarity metrics
    rouge_score = calculate_rouge_score(llm_answer, ground_truth_answer)
    jaccard_index = calculate_jaccard_index(llm_answer, ground_truth_answer)
    levenshtein_dist = distance(llm_answer, ground_truth_answer)
    
    # Normalize Levenshtein distance
    max_len = max(len(llm_answer), len(ground_truth_answer))
    normalized_levenshtein = 1.0 - (levenshtein_dist / max_len) if max_len > 0 else 0.0
    
    return {
        'question': question,
        'ground_truth': ground_truth_answer,
        'llm_answer': llm_answer,
        'rouge_score': rouge_score,
        'jaccard_index': jaccard_index,
        'levenshtein_distance': levenshtein_dist,
        'normalized_levenshtein': normalized_levenshtein,
    }


def run_knowledge_qa_evaluation(
    qa_pairs: List[Dict[str, str]],
    api_key: str,
    model_choice: str,
    provider: str,
    num_runs: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
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
        
    Returns:
        List of evaluation results for each run
    """
    
    all_results = []
    
    for run_idx in range(num_runs):
        run_results = []
        
        for qa_pair in qa_pairs:
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
            )
            
            # Skip if error
            if isinstance(llm_answer, str) and llm_answer.startswith("Error"):
                continue
            
            # Evaluate comparison
            evaluation = evaluate_qa_comparison(
                question,
                ground_truth,
                llm_answer,
            )
            
            run_results.append(evaluation)
        
        all_results.append(run_results)
    
    return all_results


def calculate_aggregate_metrics(
    all_results: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all runs.
    
    Args:
        all_results: List of evaluation results for each run
        
    Returns:
        Dictionary with aggregate statistics
    """
    
    if not all_results or not all_results[0]:
        return {}
    
    # Flatten all results
    all_evals = [eval_item for run in all_results for eval_item in run]
    
    # Calculate averages
    avg_rouge = sum(e['rouge_score'] for e in all_evals) / len(all_evals)
    avg_jaccard = sum(e['jaccard_index'] for e in all_evals) / len(all_evals)
    avg_levenshtein = sum(e['levenshtein_distance'] for e in all_evals) / len(all_evals)
    avg_norm_levenshtein = sum(e['normalized_levenshtein'] for e in all_evals) / len(all_evals)
    
    # Calculate min/max
    rouge_scores = [e['rouge_score'] for e in all_evals]
    jaccard_scores = [e['jaccard_index'] for e in all_evals]
    
    return {
        'total_runs': len(all_results),
        'total_evaluations': len(all_evals),
        'avg_rouge_score': avg_rouge,
        'avg_jaccard_index': avg_jaccard,
        'avg_levenshtein_distance': avg_levenshtein,
        'avg_normalized_levenshtein': avg_norm_levenshtein,
        'min_rouge': min(rouge_scores),
        'max_rouge': max(rouge_scores),
        'min_jaccard': min(jaccard_scores),
        'max_jaccard': max(jaccard_scores),
    }
