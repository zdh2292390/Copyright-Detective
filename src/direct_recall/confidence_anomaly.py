"""Confidence Anomaly Detection for Black-Box Copyright Detection.

This module implements a "confidence anomaly detector" that monitors the generation
process for abnormal confidence spikes. When the LLM produces outputs with 
consecutively high confidence peaks, it may indicate memorization of training data.

The approach is inspired by uncertainty quantification (UQ) techniques and works
as a black-box probe that doesn't require access to the training database.

Key Insight:
- If a model has memorized content, it will exhibit abnormally high confidence
  (low perplexity) when generating memorized sequences.
- Confidence spikes are detected by analyzing logprobs returned from LLM APIs.

Detection Methods:
1. **Threshold-based Detection**: Identify consecutive tokens above a confidence threshold.
2. **Z-Score Outlier Detection**: Detect tokens that are statistical outliers.
3. **Sliding Window Analysis**: Detect local confidence anomalies within windows.
4. **Entropy Analysis**: Low entropy sequences indicate predictable (possibly memorized) content.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import openai
import anthropic
import google.generativeai as genai

from src.common.progress import (
    start_llm_progress,
    update_llm_progress,
    complete_llm_progress,
)


@dataclass
class TokenLogprob:
    """Represents a token with its log probability information."""
    token: str
    logprob: float
    linear_prob: float  # exp(logprob)
    
    # Token classification
    is_punctuation: bool = False
    is_common_word: bool = False
    is_whitespace: bool = False
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this token has high confidence (>90%)."""
        return self.linear_prob > 0.9
    
    @property
    def is_very_high_confidence(self) -> bool:
        """Check if this token has very high confidence (>95%)."""
        return self.linear_prob > 0.95
    
    @property
    def entropy_contribution(self) -> float:
        """Calculate the entropy contribution of this token (in bits).
        
        Lower values indicate more predictable (higher confidence) tokens.
        """
        if self.linear_prob <= 0 or self.linear_prob >= 1:
            return 0.0
        return -self.linear_prob * math.log2(self.linear_prob)


# Common English words that typically have high confidence (not indicative of memorization)
COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "to", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "yes", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
}


@dataclass
class ConfidenceSpike:
    """Represents a detected confidence spike in the generation."""
    start_index: int
    end_index: int
    tokens: List[TokenLogprob]
    avg_confidence: float
    max_confidence: float
    text: str
    
    # Spike quality metrics
    intensity_score: float = 0.0  # How "intense" the spike is (based on confidence levels)
    
    @property
    def length(self) -> int:
        return self.end_index - self.start_index
    
    def calculate_intensity(self) -> float:
        """Calculate spike intensity based on confidence levels above threshold.
        
        Higher intensity = more tokens with very high (>95%) confidence.
        """
        if not self.tokens:
            return 0.0
        very_high_count = sum(1 for t in self.tokens if t.linear_prob > 0.95)
        return very_high_count / len(self.tokens)
    

@dataclass
class ConfidenceAnalysisResult:
    """Complete result of confidence anomaly analysis."""
    tokens: List[TokenLogprob]
    spikes: List[ConfidenceSpike]
    overall_avg_confidence: float
    overall_std_confidence: float
    memorization_score: float  # 0-1 score indicating likelihood of memorization
    generated_text: str
    analysis_available: bool  # Whether logprobs were available
    error_message: Optional[str] = None

    # Statistics
    high_confidence_ratio: float = 0.0  # Ratio of tokens with >90% confidence
    spike_coverage: float = 0.0  # Ratio of text covered by spikes
    longest_spike_length: int = 0
    
    # Enhanced metrics
    avg_entropy: float = 0.0  # Average entropy across tokens (lower = more predictable)
    perplexity: float = 0.0  # Perplexity of the generation
    rare_token_confidence: float = 0.0  # Avg confidence of non-common tokens
    zscore_outliers: int = 0  # Number of statistical outliers detected
    consecutive_spike_bonus: float = 0.0  # Bonus for consecutive spikes
    
    # Confidence timeline for visualization
    confidence_timeline: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_available": self.analysis_available,
            "generated_text": self.generated_text,
            "overall_avg_confidence": self.overall_avg_confidence,
            "overall_std_confidence": self.overall_std_confidence,
            "memorization_score": self.memorization_score,
            "high_confidence_ratio": self.high_confidence_ratio,
            "spike_coverage": self.spike_coverage,
            "longest_spike_length": self.longest_spike_length,
            "num_spikes": len(self.spikes),
            "total_tokens": len(self.tokens),
            # Enhanced metrics
            "avg_entropy": self.avg_entropy,
            "perplexity": self.perplexity,
            "rare_token_confidence": self.rare_token_confidence,
            "zscore_outliers": self.zscore_outliers,
            "consecutive_spike_bonus": self.consecutive_spike_bonus,
            "confidence_timeline": self.confidence_timeline,
            "spikes": [
                {
                    "start_index": spike.start_index,
                    "end_index": spike.end_index,
                    "length": spike.length,
                    "avg_confidence": spike.avg_confidence,
                    "max_confidence": spike.max_confidence,
                    "text": spike.text,
                    "intensity_score": spike.intensity_score,
                }
                for spike in self.spikes
            ],
            "error_message": self.error_message,
        }


def _classify_token(token: TokenLogprob) -> TokenLogprob:
    """Classify a token as punctuation, common word, whitespace, etc."""
    text = token.token.strip().lower()
    
    # Check if whitespace
    if not token.token.strip():
        token.is_whitespace = True
    # Check if punctuation
    elif all(c in '.,;:!?"\'-()[]{}' for c in token.token.strip()):
        token.is_punctuation = True
    # Check if common word
    elif text in COMMON_WORDS:
        token.is_common_word = True
    
    return token


def _calculate_entropy(tokens: List[TokenLogprob]) -> float:
    """Calculate average entropy across tokens (in bits).
    
    Lower entropy indicates more predictable/deterministic generation,
    which may suggest memorization.
    """
    if not tokens:
        return 0.0
    
    entropies = [t.entropy_contribution for t in tokens]
    return statistics.mean(entropies)


def _calculate_perplexity(tokens: List[TokenLogprob]) -> float:
    """Calculate perplexity of the generated sequence.
    
    Perplexity = exp(-1/N * sum(log_probs))
    Lower perplexity suggests more confident/predictable generation.
    """
    if not tokens:
        return float('inf')
    
    avg_logprob = statistics.mean(t.logprob for t in tokens)
    return math.exp(-avg_logprob)


def _detect_zscore_outliers(
    tokens: List[TokenLogprob],
    zscore_threshold: float = 2.0,
) -> List[int]:
    """Detect tokens with unusually high confidence using z-score method.
    
    Args:
        tokens: List of tokens with logprob information.
        zscore_threshold: Number of standard deviations above mean to consider outlier.
    
    Returns:
        List of indices of outlier tokens.
    """
    if len(tokens) < 3:
        return []
    
    probs = [t.linear_prob for t in tokens]
    mean_prob = statistics.mean(probs)
    std_prob = statistics.stdev(probs)
    
    if std_prob == 0:
        return []
    
    outliers = []
    for i, prob in enumerate(probs):
        zscore = (prob - mean_prob) / std_prob
        if zscore > zscore_threshold:
            outliers.append(i)
    
    return outliers


def _sliding_window_spike_detection(
    tokens: List[TokenLogprob],
    window_size: int = 5,
    threshold_ratio: float = 0.8,
    min_avg_confidence: float = 0.85,
) -> List[ConfidenceSpike]:
    """Detect spikes using sliding window analysis.
    
    This method finds windows where a high proportion of tokens exceed the confidence
    threshold, allowing for occasional dips within otherwise high-confidence sequences.
    
    Args:
        tokens: List of tokens with logprob information.
        window_size: Size of the sliding window.
        threshold_ratio: Minimum ratio of high-confidence tokens in window.
        min_avg_confidence: Minimum average confidence for window to be spike.
    
    Returns:
        List of detected confidence spikes.
    """
    if len(tokens) < window_size:
        return []
    
    spikes: List[ConfidenceSpike] = []
    in_spike = False
    spike_start = 0
    spike_tokens: List[TokenLogprob] = []
    
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        high_conf_count = sum(1 for t in window if t.linear_prob >= 0.85)
        avg_conf = statistics.mean(t.linear_prob for t in window)
        
        is_spike_window = (
            high_conf_count / window_size >= threshold_ratio and
            avg_conf >= min_avg_confidence
        )
        
        if is_spike_window:
            if not in_spike:
                in_spike = True
                spike_start = i
                spike_tokens = list(window)
            else:
                # Extend spike
                spike_tokens.append(tokens[i + window_size - 1])
        else:
            if in_spike and len(spike_tokens) >= 3:
                spike_text = "".join(t.token for t in spike_tokens)
                avg_c = statistics.mean(t.linear_prob for t in spike_tokens)
                max_c = max(t.linear_prob for t in spike_tokens)
                
                spikes.append(ConfidenceSpike(
                    start_index=spike_start,
                    end_index=spike_start + len(spike_tokens),
                    tokens=spike_tokens,
                    avg_confidence=avg_c,
                    max_confidence=max_c,
                    text=spike_text,
                    detection_method="window",
                ))
            in_spike = False
            spike_tokens = []
    
    # Handle spike at end
    if in_spike and len(spike_tokens) >= 3:
        spike_text = "".join(t.token for t in spike_tokens)
        avg_c = statistics.mean(t.linear_prob for t in spike_tokens)
        max_c = max(t.linear_prob for t in spike_tokens)
        
        spikes.append(ConfidenceSpike(
            start_index=spike_start,
            end_index=spike_start + len(spike_tokens),
            tokens=spike_tokens,
            avg_confidence=avg_c,
            max_confidence=max_c,
            text=spike_text,
            detection_method="window",
        ))
    
    return spikes


def _detect_confidence_spikes(
    tokens: List[TokenLogprob],
    confidence_threshold: float = 0.85,
    min_spike_length: int = 3,
) -> List[ConfidenceSpike]:
    """Detect consecutive sequences of high-confidence tokens.
    
    Args:
        tokens: List of tokens with logprob information.
        confidence_threshold: Minimum linear probability to consider "high confidence".
        min_spike_length: Minimum number of consecutive high-confidence tokens to be a spike.
    
    Returns:
        List of detected confidence spikes.
    """
    spikes: List[ConfidenceSpike] = []
    
    if not tokens:
        return spikes
    
    current_spike_start: Optional[int] = None
    current_spike_tokens: List[TokenLogprob] = []
    
    for i, token in enumerate(tokens):
        is_high = token.linear_prob >= confidence_threshold
        
        if is_high:
            if current_spike_start is None:
                current_spike_start = i
                current_spike_tokens = [token]
            else:
                current_spike_tokens.append(token)
        else:
            # End of potential spike
            if current_spike_start is not None and len(current_spike_tokens) >= min_spike_length:
                spike_text = "".join(t.token for t in current_spike_tokens)
                avg_conf = statistics.mean(t.linear_prob for t in current_spike_tokens)
                max_conf = max(t.linear_prob for t in current_spike_tokens)
                
                spikes.append(ConfidenceSpike(
                    start_index=current_spike_start,
                    end_index=i,
                    tokens=current_spike_tokens.copy(),
                    avg_confidence=avg_conf,
                    max_confidence=max_conf,
                    text=spike_text,
                ))
            
            current_spike_start = None
            current_spike_tokens = []
    
    # Check for spike at the end
    if current_spike_start is not None and len(current_spike_tokens) >= min_spike_length:
        spike_text = "".join(t.token for t in current_spike_tokens)
        avg_conf = statistics.mean(t.linear_prob for t in current_spike_tokens)
        max_conf = max(t.linear_prob for t in current_spike_tokens)
        
        spike = ConfidenceSpike(
            start_index=current_spike_start,
            end_index=len(tokens),
            tokens=current_spike_tokens,
            avg_confidence=avg_conf,
            max_confidence=max_conf,
            text=spike_text,
            detection_method="threshold",
        )
        spike.intensity_score = spike.calculate_intensity()
        spikes.append(spike)
    
    # Calculate intensity for all spikes
    for spike in spikes:
        spike.intensity_score = spike.calculate_intensity()
    
    return spikes


def _merge_overlapping_spikes(spikes: List[ConfidenceSpike]) -> List[ConfidenceSpike]:
    """Merge overlapping or adjacent spikes from different detection methods."""
    if not spikes:
        return []
    
    # Sort by start index
    sorted_spikes = sorted(spikes, key=lambda s: s.start_index)
    merged: List[ConfidenceSpike] = []
    
    current = sorted_spikes[0]
    for spike in sorted_spikes[1:]:
        # Check for overlap or adjacency (within 2 tokens)
        if spike.start_index <= current.end_index + 2:
            # Merge: extend current spike
            new_end = max(current.end_index, spike.end_index)
            all_tokens = current.tokens + [t for t in spike.tokens if t not in current.tokens]
            
            current = ConfidenceSpike(
                start_index=current.start_index,
                end_index=new_end,
                tokens=all_tokens,
                avg_confidence=statistics.mean(t.linear_prob for t in all_tokens),
                max_confidence=max(t.linear_prob for t in all_tokens),
                text="".join(t.token for t in all_tokens),
                detection_method="combined",
                intensity_score=current.intensity_score,
            )
        else:
            merged.append(current)
            current = spike
    
    merged.append(current)
    return merged


def _calculate_memorization_score(
    tokens: List[TokenLogprob],
    spikes: List[ConfidenceSpike],
    rare_token_confidence: float = 0.0,
    consecutive_spike_bonus: float = 0.0,
) -> Tuple[float, float, float, int]:
    """Calculate a memorization score based on confidence patterns.
    
    Returns:
        Tuple of (memorization_score, high_confidence_ratio, spike_coverage, longest_spike_length)
    """
    if not tokens:
        return 0.0, 0.0, 0.0, 0
    
    # High confidence ratio (>90% confidence)
    high_conf_count = sum(1 for t in tokens if t.linear_prob > 0.9)
    high_confidence_ratio = high_conf_count / len(tokens)
    
    # Spike coverage
    spike_token_count = sum(spike.length for spike in spikes)
    spike_coverage = spike_token_count / len(tokens) if tokens else 0.0
    
    # Longest spike
    longest_spike_length = max((spike.length for spike in spikes), default=0)
    
    # Calculate spike intensity bonus (high-intensity spikes are more indicative)
    spike_intensity = 0.0
    if spikes:
        spike_intensity = statistics.mean(s.intensity_score for s in spikes)
    
    # Calculate rare token confidence impact
    # High confidence on rare tokens is very indicative of memorization
    rare_confidence_factor = min(rare_token_confidence * 1.2, 1.0)  # Amplify but cap at 1
    
    # Combined memorization score (weighted combination)
    # - High confidence ratio: weight 0.2 (base measure)
    # - Spike coverage: weight 0.25 (how much of output is in spikes)
    # - Normalized longest spike (capped at 20 tokens): weight 0.2
    # - Spike intensity: weight 0.15 (quality of spikes)
    # - Rare token confidence: weight 0.15 (memorization of specific content)
    # - Consecutive spike bonus: weight 0.05 (pattern recognition)
    normalized_longest = min(longest_spike_length / 20.0, 1.0)
    
    memorization_score = (
        0.20 * high_confidence_ratio +
        0.25 * spike_coverage +
        0.20 * normalized_longest +
        0.15 * spike_intensity +
        0.15 * rare_confidence_factor +
        0.05 * consecutive_spike_bonus
    )
    
    return memorization_score, high_confidence_ratio, spike_coverage, longest_spike_length


def get_completion_with_logprobs_openai(
    prompt: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 500,
    base_url: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[TokenLogprob], Optional[str]]:
    """Get completion with logprobs from OpenAI-compatible API.
    
    Returns:
        Tuple of (generated_text, token_logprobs, error_message)
    """
    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = openai.OpenAI(**client_kwargs)
        
        request_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        }
        
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers
        
        response = client.chat.completions.create(**request_kwargs)
        
        generated_text = response.choices[0].message.content or ""
        
        # Extract logprobs
        tokens: List[TokenLogprob] = []
        logprobs_content = response.choices[0].logprobs
        
        if logprobs_content and hasattr(logprobs_content, 'content') and logprobs_content.content:
            for token_info in logprobs_content.content:
                logprob = token_info.logprob
                linear_prob = math.exp(logprob) if logprob > -100 else 0.0
                tokens.append(TokenLogprob(
                    token=token_info.token,
                    logprob=logprob,
                    linear_prob=linear_prob,
                ))
        
        return generated_text.strip(), tokens, None
        
    except Exception as e:
        return "", [], f"Error calling API: {str(e)}"


def run_confidence_anomaly_detection(
    prompt: str,
    api_key: str,
    model_name: str,
    provider: str = "OpenAI",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 500,
    confidence_threshold: float = 0.85,
    min_spike_length: int = 3,
    progress_message: Optional[str] = None,
) -> ConfidenceAnalysisResult:
    """Run confidence anomaly detection on LLM generation.
    
    This function generates text and analyzes the confidence patterns in the
    generation to detect potential memorization.
    
    Args:
        prompt: The input prompt for generation.
        api_key: API key for the LLM provider.
        model_name: Name of the model to use.
        provider: LLM provider ("OpenAI", "OpenRouter").
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens to generate.
        confidence_threshold: Threshold for detecting high confidence tokens.
        min_spike_length: Minimum consecutive high-confidence tokens for a spike.
        progress_message: Optional progress message.
    
    Returns:
        ConfidenceAnalysisResult with analysis details.
    """
    label_placeholder, bar_placeholder, progress_bar = start_llm_progress(
        progress_message or f"Running confidence analysis · {model_name}"
    )
    update_llm_progress(progress_bar, value=15)
    
    # Currently only OpenAI and OpenRouter support logprobs
    if provider not in ("OpenAI", "OpenRouter"):
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message=f"Logprobs not available for {provider}",
            success=False,
            linger=0.5,
        )
        return ConfidenceAnalysisResult(
            tokens=[],
            spikes=[],
            overall_avg_confidence=0.0,
            overall_std_confidence=0.0,
            memorization_score=0.0,
            generated_text="",
            analysis_available=False,
            error_message=f"Confidence analysis requires logprobs, which are not available for {provider}. "
                         "This feature works with OpenAI and OpenRouter providers.",
        )
    
    update_llm_progress(progress_bar, value=30)
    
    # Get completion with logprobs
    if provider == "OpenRouter":
        generated_text, tokens, error = get_completion_with_logprobs_openai(
            prompt=prompt,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Copyright Detective"
            },
        )
    else:  # OpenAI
        generated_text, tokens, error = get_completion_with_logprobs_openai(
            prompt=prompt,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    update_llm_progress(progress_bar, value=70)
    
    if error:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="Confidence analysis failed",
            success=False,
            linger=0.5,
        )
        return ConfidenceAnalysisResult(
            tokens=[],
            spikes=[],
            overall_avg_confidence=0.0,
            overall_std_confidence=0.0,
            memorization_score=0.0,
            generated_text=generated_text,
            analysis_available=False,
            error_message=error,
        )
    
    # Check if logprobs were returned
    if not tokens:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="No logprobs returned by API",
            success=False,
            linger=0.5,
        )
        return ConfidenceAnalysisResult(
            tokens=[],
            spikes=[],
            overall_avg_confidence=0.0,
            overall_std_confidence=0.0,
            memorization_score=0.0,
            generated_text=generated_text,
            analysis_available=False,
            error_message="The API did not return logprobs. This may be due to model limitations or API settings.",
        )
    
    update_llm_progress(progress_bar, value=80)
    
    # Classify tokens
    for token in tokens:
        _classify_token(token)
    
    # Calculate overall statistics
    linear_probs = [t.linear_prob for t in tokens]
    overall_avg = statistics.mean(linear_probs)
    overall_std = statistics.stdev(linear_probs) if len(linear_probs) > 1 else 0.0
    
    # Calculate entropy and perplexity
    avg_entropy = _calculate_entropy(tokens)
    perplexity = _calculate_perplexity(tokens)
    
    # Build confidence timeline
    confidence_timeline = [t.linear_prob for t in tokens]
    
    # Detect spikes using threshold method
    threshold_spikes = _detect_confidence_spikes(
        tokens,
        confidence_threshold=confidence_threshold,
        min_spike_length=min_spike_length,
    )
    
    # Detect spikes using sliding window method
    window_spikes = _sliding_window_spike_detection(tokens)
    
    # Detect statistical outliers
    zscore_outliers = _detect_zscore_outliers(tokens)
    
    # Merge spikes from different methods
    all_spikes = threshold_spikes + window_spikes
    spikes = _merge_overlapping_spikes(all_spikes)
    
    update_llm_progress(progress_bar, value=90)
    
    # Calculate rare token confidence
    rare_tokens = [t for t in tokens if not t.is_common_word and not t.is_punctuation and not t.is_whitespace]
    rare_token_confidence = statistics.mean(t.linear_prob for t in rare_tokens) if rare_tokens else 0.0
    
    # Calculate consecutive spike bonus (spikes close together)
    consecutive_spike_bonus = 0.0
    if len(spikes) >= 2:
        gaps = []
        for i in range(len(spikes) - 1):
            gap = spikes[i + 1].start_index - spikes[i].end_index
            gaps.append(gap)
        avg_gap = statistics.mean(gaps)
        # Bonus for small gaps (consecutive spikes)
        consecutive_spike_bonus = max(0.0, 1.0 - (avg_gap / 20.0))  # Full bonus if gap < 1, zero if gap >= 20
    
    # Calculate memorization score
    mem_score, high_conf_ratio, spike_cov, longest_spike = _calculate_memorization_score(
        tokens, spikes, rare_token_confidence, consecutive_spike_bonus
    )
    
    complete_llm_progress(
        label_placeholder,
        bar_placeholder,
        progress_bar,
        final_message=f"Confidence analysis complete · {len(spikes)} spikes detected",
        success=True,
    )
    
    return ConfidenceAnalysisResult(
        tokens=tokens,
        spikes=spikes,
        overall_avg_confidence=overall_avg,
        overall_std_confidence=overall_std,
        memorization_score=mem_score,
        generated_text=generated_text,
        analysis_available=True,
        high_confidence_ratio=high_conf_ratio,
        spike_coverage=spike_cov,
        longest_spike_length=longest_spike,
        avg_entropy=avg_entropy,
        perplexity=perplexity,
        rare_token_confidence=rare_token_confidence,
        zscore_outliers=len(zscore_outliers),
        consecutive_spike_bonus=consecutive_spike_bonus,
        confidence_timeline=confidence_timeline,
    )


def format_confidence_analysis_summary(result: ConfidenceAnalysisResult) -> str:
    """Format the confidence analysis result as a human-readable summary."""
    if not result.analysis_available:
        return f"⚠️ Confidence analysis unavailable: {result.error_message}"
    
    lines = [
        "📊 **Confidence Anomaly Analysis**",
        "",
        "**Core Metrics:**",
        f"- **Memorization Score**: {result.memorization_score:.2%}",
        f"- **Average Confidence**: {result.overall_avg_confidence:.2%}",
        f"- **Confidence Std Dev**: {result.overall_std_confidence:.2%}",
        f"- **High Confidence Token Ratio** (>90%): {result.high_confidence_ratio:.2%}",
        "",
        "**Spike Analysis:**",
        f"- **Number of Spikes**: {len(result.spikes)}",
        f"- **Spike Coverage**: {result.spike_coverage:.2%}",
        f"- **Longest Spike**: {result.longest_spike_length} tokens",
        "",
        "**Advanced Metrics:**",
        f"- **Average Entropy**: {result.avg_entropy:.4f} bits",
        f"- **Perplexity**: {result.perplexity:.2f}",
        f"- **Rare Token Confidence**: {result.rare_token_confidence:.2%}",
        f"- **Z-Score Outliers**: {result.zscore_outliers}",
        "",
    ]
    
    if result.spikes:
        lines.append("**Detected Confidence Spikes:**")
        for i, spike in enumerate(result.spikes[:5], 1):  # Show top 5 spikes
            spike_text = spike.text[:50] + "..." if len(spike.text) > 50 else spike.text
            method_tag = f"[{spike.detection_method}]" if spike.detection_method != "threshold" else ""
            lines.append(
                f"  {i}. {method_tag}\"{spike_text}\" "
                f"(len: {spike.length}, avg: {spike.avg_confidence:.2%}, intensity: {spike.intensity_score:.2%})"
            )
        if len(result.spikes) > 5:
            lines.append(f"  ... and {len(result.spikes) - 5} more spikes")
    
    # Interpretation
    lines.append("")
    lines.append("**Interpretation:**")
    if result.memorization_score > 0.7:
        lines.append("🚨 **High memorization likelihood detected!** The model shows strong confidence patterns consistent with verbatim memorization.")
        lines.append("   - Multiple long high-confidence sequences detected")
        lines.append("   - High confidence on rare/specific tokens suggests trained content")
    elif result.memorization_score > 0.4:
        lines.append("⚠️ **Moderate memorization signals.** Some confidence patterns suggest potential memorization.")
        lines.append("   - Some high-confidence spikes detected")
        lines.append("   - Consider comparing with other analysis methods")
    else:
        lines.append("✅ **Low memorization likelihood.** Confidence patterns appear normal for generated content.")
        lines.append("   - No significant high-confidence spike patterns")
        lines.append("   - Natural variation in confidence levels observed")
    
    return "\n".join(lines)


def generate_confidence_visualization_data(result: ConfidenceAnalysisResult) -> Dict[str, Any]:
    """Generate data for confidence timeline visualization.
    
    Returns a dictionary with data suitable for plotting.
    """
    if not result.analysis_available:
        return {"error": result.error_message}
    
    # Build spike regions for highlighting
    spike_regions = []
    for spike in result.spikes:
        spike_regions.append({
            "start": spike.start_index,
            "end": spike.end_index,
            "avg_confidence": spike.avg_confidence,
            "method": spike.detection_method,
        })
    
    return {
        "timeline": result.confidence_timeline,
        "avg_confidence": result.overall_avg_confidence,
        "threshold_line": 0.85,  # Default threshold
        "spike_regions": spike_regions,
        "total_tokens": len(result.tokens),
        "memorization_score": result.memorization_score,
    }


def analyze_logprobs_for_confidence(
    logprobs_data: List[Dict[str, Any]],
    generated_text: str,
    confidence_threshold: float = 0.85,
    min_spike_length: int = 3,
) -> ConfidenceAnalysisResult:
    """Analyze pre-existing logprobs data for confidence anomalies.
    
    This function allows analyzing logprobs that were already obtained from an LLM call,
    avoiding the need for a separate API call.
    
    Args:
        logprobs_data: List of dicts with 'token', 'logprob', and 'linear_prob' keys.
        generated_text: The generated text corresponding to the logprobs.
        confidence_threshold: Threshold for detecting high confidence tokens.
        min_spike_length: Minimum consecutive high-confidence tokens for a spike.
    
    Returns:
        ConfidenceAnalysisResult with analysis details.
    """
    if not logprobs_data:
        return ConfidenceAnalysisResult(
            tokens=[],
            spikes=[],
            overall_avg_confidence=0.0,
            overall_std_confidence=0.0,
            memorization_score=0.0,
            generated_text=generated_text,
            analysis_available=False,
            error_message="No logprobs data available. This may be due to model limitations or API settings.",
        )
    
    # Convert logprobs data to TokenLogprob objects
    tokens: List[TokenLogprob] = []
    for item in logprobs_data:
        token = TokenLogprob(
            token=item.get("token", ""),
            logprob=item.get("logprob", 0.0),
            linear_prob=item.get("linear_prob", 0.0),
        )
        _classify_token(token)
        tokens.append(token)
    
    # Calculate overall statistics
    linear_probs = [t.linear_prob for t in tokens]
    overall_avg = statistics.mean(linear_probs) if linear_probs else 0.0
    overall_std = statistics.stdev(linear_probs) if len(linear_probs) > 1 else 0.0
    
    # Calculate entropy and perplexity
    avg_entropy = _calculate_entropy(tokens)
    perplexity = _calculate_perplexity(tokens)
    
    # Build confidence timeline
    confidence_timeline = [t.linear_prob for t in tokens]
    
    # Detect spikes using multiple methods
    threshold_spikes = _detect_confidence_spikes(tokens, confidence_threshold, min_spike_length)
    window_spikes = _sliding_window_spike_detection(tokens)
    
    # Merge all spikes
    all_spikes = threshold_spikes + window_spikes
    merged_spikes = _merge_overlapping_spikes(all_spikes)
    
    # Detect z-score outliers
    zscore_outlier_indices = _detect_zscore_outliers(tokens)
    
    # Calculate rare token confidence (non-common words)
    rare_tokens = [t for t in tokens if not t.is_common_word and not t.is_punctuation and not t.is_whitespace]
    rare_token_confidence = statistics.mean(t.linear_prob for t in rare_tokens) if rare_tokens else 0.0
    
    # Calculate consecutive spike bonus
    consecutive_spike_bonus = 0.0
    if len(merged_spikes) >= 2:
        # Check for consecutive spikes (close together)
        consecutive_count = 0
        for i in range(len(merged_spikes) - 1):
            gap = merged_spikes[i + 1].start_index - merged_spikes[i].end_index
            if gap <= 5:  # Within 5 tokens
                consecutive_count += 1
        consecutive_spike_bonus = min(consecutive_count * 0.1, 0.3)  # Cap at 0.3
    
    # Calculate memorization score
    memorization_score, high_conf_ratio, spike_coverage, longest_spike = _calculate_memorization_score(
        tokens, merged_spikes, rare_token_confidence, consecutive_spike_bonus
    )
    
    return ConfidenceAnalysisResult(
        tokens=tokens,
        spikes=merged_spikes,
        overall_avg_confidence=overall_avg,
        overall_std_confidence=overall_std,
        memorization_score=memorization_score,
        generated_text=generated_text,
        analysis_available=True,
        high_confidence_ratio=high_conf_ratio,
        spike_coverage=spike_coverage,
        longest_spike_length=longest_spike,
        avg_entropy=avg_entropy,
        perplexity=perplexity,
        rare_token_confidence=rare_token_confidence,
        zscore_outliers=len(zscore_outlier_indices),
        consecutive_spike_bonus=consecutive_spike_bonus,
        confidence_timeline=confidence_timeline,
    )
