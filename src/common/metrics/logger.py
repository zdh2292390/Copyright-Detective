import string
import re
from collections import Counter
from typing import Dict, Any, Optional, List


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
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
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


class FactRecallLogger:
    """
    Evaluate Open-ended Questions using Token-level F1 Score.
    Used for Fact Recall evaluation method.
    """
    
    def __init__(self):
        self.entries = []

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
        self.entries.append(entry)

    def report(self) -> Dict[str, Any]:
        """
        Generate an evaluation report.
        
        Returns:
            A dictionary containing mean metrics and all entries.
        """
        if not self.entries:
            return {
                'mean_precision': 0.0,
                'mean_recall': 0.0,
                'mean_f1': 0.0,
                'entries': []
            }
        
        precision_scores = [e['precision'] for e in self.entries]
        recall_scores = [e['recall'] for e in self.entries]
        f1_scores = [e['f1'] for e in self.entries]
        
        return {
            'mean_precision': sum(precision_scores) / len(precision_scores),
            'mean_recall': sum(recall_scores) / len(recall_scores),
            'mean_f1': sum(f1_scores) / len(f1_scores),
            'entries': self.entries
        }


# Keep RougeEvalLogger alias for backward compatibility
# Now uses Token-level F1 Score instead of ROUGE
RougeEvalLogger = FactRecallLogger
