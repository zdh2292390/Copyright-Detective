import string
import re
from collections import Counter
from typing import Dict, Any, Optional, List, Callable

# LLM Judge evaluation prompt template
LLM_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to assess whether the model's answer correctly captures the key information from the ground truth answer.

**Question:**
{question}

**Ground Truth Answer:**
{ground_truth}

**Model's Answer:**
{prediction}

**Evaluation Criteria:**
1. **Factual Accuracy**: Does the model's answer contain the correct facts from the ground truth?
2. **Completeness**: Does the model's answer cover the key information in the ground truth?
3. **Relevance**: Is the model's answer relevant to the question?

**Instructions:**
- Compare the model's answer with the ground truth answer.
- Consider semantic similarity, not just exact word matching.
- Minor wording differences are acceptable if the meaning is preserved.
- Provide a score from 0 to 1 (where 0 = completely wrong, 1 = perfectly correct).

**Output Format:**
Return ONLY a JSON object with the following format (no other text):
{{"score": <float between 0 and 1>, "reasoning": "<brief explanation>"}}"""


def normalize_answer(s: str) -> str:
    """
    Normalize the answer for evaluation:
    1. Lowercasing
    2. Punctuation removal
    3. Article removal (a, an, the)
    4. Whitespace normalization
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    
    # 1. Lowercasing
    s = s.lower()
    
    # 2. Punctuation removal
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Article removal (a, an, the)
    # Use regex to match standalone article words
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # 4. Whitespace normalization - merge multiple spaces into one and strip
    s = ' '.join(s.split())
    
    return s


def get_tokens(s: str) -> List[str]:
    """Split the normalized text into a list of tokens (words)."""
    if not s:
        return []
    return s.split()


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute Token-level F1 Score.
    
    Args:
        prediction: The model-generated answer.
        ground_truth: The reference answer.
    
    Returns:
        A dictionary containing precision, recall, and f1.
    """
    # Normalize
    pred_normalized = normalize_answer(prediction)
    truth_normalized = normalize_answer(ground_truth)
    
    # Get tokens
    pred_tokens = get_tokens(pred_normalized)
    truth_tokens = get_tokens(truth_normalized)
    
    # Handle edge cases
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Use Counter for multiset intersection (considering word frequency)
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    
    # Compute common tokens (multiset intersection)
    common = pred_counter & truth_counter
    num_same = sum(common.values())
    
    # Compute Precision
    precision = num_same / len(pred_tokens)
    
    # Compute Recall
    recall = num_same / len(truth_tokens)
    
    # Compute F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def parse_llm_judge_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM judge response to extract score and reasoning.
    
    Args:
        response: The raw response from the LLM judge.
    
    Returns:
        A dictionary containing score and reasoning.
    """
    import json
    
    try:
        # Try to parse as JSON directly
        result = json.loads(response.strip())
        score = float(result.get('score', 0.0))
        reasoning = result.get('reasoning', '')
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        return {'score': score, 'reasoning': reasoning}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Try to extract JSON from the response
    json_pattern = r'\{[^{}]*"score"[^{}]*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            score = float(result.get('score', 0.0))
            reasoning = result.get('reasoning', '')
            score = max(0.0, min(1.0, score))
            return {'score': score, 'reasoning': reasoning}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Try to extract just the score using regex
    score_pattern = r'"?score"?\s*[:\s]+\s*([0-9]*\.?[0-9]+)'
    score_match = re.search(score_pattern, response, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
            return {'score': score, 'reasoning': 'Score extracted from response'}
        except ValueError:
            pass
    
    # Default fallback
    return {'score': 0.0, 'reasoning': 'Failed to parse LLM judge response'}


def llm_judge_evaluate(
    question: str,
    ground_truth: str,
    prediction: str,
    llm_call_fn: Callable[[str], str],
) -> Dict[str, Any]:
    """
    Evaluate model's answer using LLM as a Judge.
    
    Args:
        question: The original question.
        ground_truth: The reference answer.
        prediction: The model-generated answer.
        llm_call_fn: A callable that takes a prompt string and returns the LLM response.
                     This should be an independent model from the one being evaluated.
    
    Returns:
        A dictionary containing:
            - score: float between 0 and 1
            - reasoning: explanation from the judge
    """
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        question=question or "N/A",
        ground_truth=ground_truth,
        prediction=prediction,
    )
    
    try:
        response = llm_call_fn(prompt)
        result = parse_llm_judge_response(response)
        return result
    except Exception as e:
        return {
            'score': 0.0,
            'reasoning': f'LLM Judge evaluation failed: {str(e)}'
        }


class FactRecallLogger:
    """
    Evaluate Open-ended Questions using Token-level F1 Score and optional LLM as a Judge.
    Used for Fact Recall evaluation method.
    """
    
    def __init__(self, llm_judge_fn: Optional[Callable[[str], str]] = None):
        """
        Initialize the FactRecallLogger.
        
        Args:
            llm_judge_fn: Optional callable for LLM-as-a-Judge evaluation.
                          Should take a prompt string and return the LLM response.
                          Use an independent model from the one being evaluated.
        """
        self.entries = []
        self.llm_judge_fn = llm_judge_fn
        self.use_llm_judge = llm_judge_fn is not None

    def log(self, prompt: str, gt: str, pred: str, question: Optional[str] = None):
        """
        Log an evaluation result.
        
        Args:
            prompt: The complete prompt.
            gt: The ground truth answer.
            pred: The model-generated prediction.
            question: The original question (optional).
        """
        # Compute Token-level F1 Score
        scores = compute_token_f1(pred, gt)
        
        entry = {
            'prompt': prompt,
            'gt': gt,
            'pred': pred,
            'question': question,
            'precision': scores['precision'],
            'recall': scores['recall'],
            'f1': scores['f1'],
        }
        
        # If LLM Judge is enabled, evaluate using LLM
        if self.use_llm_judge and self.llm_judge_fn is not None:
            judge_result = llm_judge_evaluate(
                question=question or "",
                ground_truth=gt,
                prediction=pred,
                llm_call_fn=self.llm_judge_fn,
            )
            entry['llm_judge_score'] = judge_result['score']
            entry['llm_judge_reasoning'] = judge_result['reasoning']
        
        self.entries.append(entry)

    def report(self) -> Dict[str, Any]:
        """
        Generate an evaluation report.
        
        Returns:
            A dictionary containing mean metrics and all entries.
        """
        if not self.entries:
            result = {
                'mean_precision': 0.0,
                'mean_recall': 0.0,
                'mean_f1': 0.0,
                'entries': []
            }
            if self.use_llm_judge:
                result['mean_llm_judge_score'] = 0.0
            return result
        
        precision_scores = [e['precision'] for e in self.entries]
        recall_scores = [e['recall'] for e in self.entries]
        f1_scores = [e['f1'] for e in self.entries]
        
        result = {
            'mean_precision': sum(precision_scores) / len(precision_scores),
            'mean_recall': sum(recall_scores) / len(recall_scores),
            'mean_f1': sum(f1_scores) / len(f1_scores),
            'entries': self.entries
        }
        
        # Include LLM Judge mean score if enabled
        if self.use_llm_judge:
            llm_judge_scores = [e.get('llm_judge_score', 0.0) for e in self.entries]
            result['mean_llm_judge_score'] = sum(llm_judge_scores) / len(llm_judge_scores)
        
        return result


# Keep RougeEvalLogger alias for backward compatibility
# Now uses Token-level F1 Score instead of ROUGE
RougeEvalLogger = FactRecallLogger
