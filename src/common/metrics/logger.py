from rouge_score import rouge_scorer
from typing import Dict, Any, Optional
import json


class RougeEvalLogger:
    def __init__(self):
        self.entries = []

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return " ".join(value.strip().split())

    def log(self, prompt: str, gt: str, pred: str, question: Optional[str] = None):
        normalized_gt = self._normalize_text(gt)
        normalized_pred = self._normalize_text(pred)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(normalized_gt, normalized_pred)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        if normalized_gt.casefold() == normalized_pred.casefold():
            rouge1 = rouge2 = rougeL = 1.0
        
        entry = {
            'prompt': prompt,
            'gt': gt,
            'pred': pred,
            'question': question,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
        }
        self.entries.append(entry)

    def report(self) -> Dict[str, Any]:
        if not self.entries:
            return {'mean_rouge1': 0.0, 'mean_rouge2': 0.0, 'mean_rougeL': 0.0}
        
        rouge1_scores = [e['rouge1'] for e in self.entries]
        rouge2_scores = [e['rouge2'] for e in self.entries]
        rougeL_scores = [e['rougeL'] for e in self.entries]
        
        return {
            'mean_rouge1': sum(rouge1_scores) / len(rouge1_scores),
            'mean_rouge2': sum(rouge2_scores) / len(rouge2_scores),
            'mean_rougeL': sum(rougeL_scores) / len(rougeL_scores),
            'entries': self.entries
        }