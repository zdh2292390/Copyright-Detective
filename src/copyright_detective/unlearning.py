from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import StatisticsError, fmean, median, stdev, variance
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import openai

try:  # pragma: no cover - openai may not expose this class in older versions
    from openai import BadRequestError
except ImportError:  # pragma: no cover - backwards compatibility
    BadRequestError = None  # type: ignore[assignment]

from .comparison import get_llm_completion

try:  # pragma: no cover - optional dependency
    from representational_toolkit.analysis import run_feature_analysis as _run_feature_analysis
except Exception:  # pragma: no cover - optional dependency
    try:  # fall back to local source layout
        from src.representational_toolkit.analysis import run_feature_analysis as _run_feature_analysis  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - optional dependency
        _run_feature_analysis = None

try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None


@dataclass(frozen=True)
class UnlearningStrategy:
    """Configuration for an unlearning detection probe."""

    id: str
    name: str
    description: str
    template: str

@dataclass(frozen=True)
class StatisticalTestOutcome:
    name: str
    statistic: Optional[float]
    pvalue: Optional[float]
    detail: Optional[str] = None


UNLEARNING_STRATEGIES: Tuple[UnlearningStrategy, ...] = (
    UnlearningStrategy(
        id="direct_question",
        name="Direct Question",
        description="Prompt the model to surface any residual knowledge in a structured audit summary.",
        template=(
            "You are auditing for residual memorisation. Provide a structured summary of everything you still recall about {target_description}.\n"
            "List key facts, distinctive wording, tone, and any uncertainties or gaps. Do not attempt to recreate or extend the original text."
        ),
    ),
    UnlearningStrategy(
        id="motivated_roleplay",
        name="Motivated Roleplay",
        description="Simulate a forensic review where the model inventories lingering fragments rather than reproducing text.",
        template=(
            "Assume the role of a compliance analyst validating that training data was removed.\n"
            "Summarise any fragments, themes, or stylistic signatures you still associate with: {target_description}.\n"
            "Report in bullet points, calling out confidence levels and explicitly noting if portions are missing or uncertain."
        ),
    ),
    UnlearningStrategy(
        id="indirect_summary",
        name="Indirect Summary",
        description="Request a detection-focused recap that highlights remembered and forgotten elements.",
        template=(
            "Produce a concise forensic summary of {target_description}, covering plot, motifs, notable phrases, and places where your memory is incomplete."
        ),
    ),
    UnlearningStrategy(
        id="translation_probe",
        name="Translation Probe",
        description="Test for lingering bilingual recall by asking for terminology rather than full passages.",
        template=(
            "List any key terms, quotations, or idioms you still connect with {target_description} and provide their English paraphrases or translations."
            " If nothing remains, explicitly state that you retain no details."
        ),
    ),
)


@dataclass(frozen=True)
class RepresentationalFeature:
    """Metadata describing a representational analysis capability."""

    id: str
    name: str
    description: str
    output_kind: Literal["directory", "file"]


@dataclass
class RepresentationalAnalysisResult:
    feature_id: str
    feature_name: str
    output_path: str
    generated_artifacts: List[str]
    warnings: List[str]
    error: Optional[str] = None

    @property
    def has_artifacts(self) -> bool:
        return bool(self.generated_artifacts)


REPRESENTATIONAL_FEATURES: Tuple[RepresentationalFeature, ...] = (
    RepresentationalFeature(
        id="fim",
        name="Fisher Information Matrix",
        description="Layer-wise Fisher diagonal comparison between a reference and updated model.",
        output_kind="directory",
    ),
    RepresentationalFeature(
        id="pca_shift",
        name="PCA Shift",
        description="Quantify how layer representations shift along principal components after unlearning.",
        output_kind="file",
    ),
    RepresentationalFeature(
        id="pca_sim",
        name="PCA Similarity",
        description="Track cosine similarity of leading principal components across layers post-unlearning.",
        output_kind="file",
    ),
    RepresentationalFeature(
        id="cka",
        name="Linear CKA",
        description="Evaluate layer-wise centered kernel alignment between reference and updated models.",
        output_kind="file",
    ),
)


def list_representational_features() -> List[RepresentationalFeature]:
    """Return representational analysis options for UI consumption."""

    return list(REPRESENTATIONAL_FEATURES)


def get_representational_feature(feature_id: str) -> Optional[RepresentationalFeature]:
    feature_id = feature_id.lower().strip()
    for feature in REPRESENTATIONAL_FEATURES:
        if feature.id == feature_id:
            return feature
    return None


def is_representational_analysis_available() -> bool:
    """Return whether optional representational toolkit dependencies are available."""

    return _run_feature_analysis is not None


def _normalise_query_inputs(query: Sequence[str]) -> List[str]:
    normalised: List[str] = []
    for item in query:
        if not item:
            continue
        cleaned = item.strip()
        if cleaned:
            normalised.append(cleaned)
    return normalised


def run_representational_analysis(
    *,
    feature: str,
    model_reference_path: str,
    model_path: str,
    query: Sequence[str],
    output_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
) -> RepresentationalAnalysisResult:
    """Execute representational analysis for model unlearning audits."""

    if not is_representational_analysis_available():
        raise RuntimeError(
            "Representational analysis requires optional dependencies (torch, transformers, scikit-learn, matplotlib)."
        )

    feature_id = feature.lower().strip()
    feature_meta = get_representational_feature(feature_id)
    if feature_meta is None:
        raise ValueError(f"Unknown representational feature: {feature}")

    if not model_reference_path or not model_reference_path.strip():
        raise ValueError("Reference model path is required for representational analysis.")
    if not model_path or not model_path.strip():
        raise ValueError("Updated model path is required for representational analysis.")

    normalised_query = _normalise_query_inputs(query)
    if not normalised_query:
        raise ValueError("At least one non-empty query string is required for representational analysis.")

    target_path = Path(output_path).expanduser()
    generated_artifacts: List[str] = []
    warnings: List[str] = []

    if feature_meta.output_kind == "directory":
        target_path.mkdir(parents=True, exist_ok=True)
    else:
        if target_path.is_dir() or not target_path.suffix:
            target_path.mkdir(parents=True, exist_ok=True)
            target_path = target_path / f"{feature_meta.id}_analysis.pdf"
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)

    before_snapshot = set()
    if feature_meta.output_kind == "directory" and target_path.exists():
        before_snapshot = {str(p.resolve()) for p in target_path.glob("*.pdf")}

    try:
        assert _run_feature_analysis is not None  # for type checkers
        _run_feature_analysis(
            feature=feature_meta.id,
            model_reference_path=model_reference_path,
            model_path=model_path,
            query=normalised_query,
            output_path=str(target_path),
            device=device,
            batch_size=batch_size,
            num_batches=num_batches,
            max_length=max_length,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency handling
        raise RuntimeError(
            f"Missing dependency for representational analysis: {exc}. Install the optional GPU/toolkit requirements and retry."
        ) from exc
    except Exception as exc:  # pragma: no cover - runtime errors
        raise RuntimeError(f"Representational analysis failed: {exc}") from exc

    if feature_meta.output_kind == "directory":
        generated_artifacts = sorted(
            str(p.resolve())
            for p in target_path.glob("*.pdf")
        )
        if before_snapshot:
            generated_artifacts = [path for path in generated_artifacts if path not in before_snapshot]
        if not generated_artifacts:
            warnings.append("Analysis completed but no PDF artifacts were detected in the output directory.")
    else:
        if target_path.exists():
            generated_artifacts = [str(target_path.resolve())]
        else:
            warnings.append("Analysis completed but the expected output file was not created.")

    return RepresentationalAnalysisResult(
        feature_id=feature_meta.id,
        feature_name=feature_meta.name,
        output_path=str(target_path.resolve() if target_path.exists() else target_path),
        generated_artifacts=generated_artifacts,
        warnings=warnings,
    )


def list_unlearning_strategies() -> List[UnlearningStrategy]:
    """Return strategies as a list for UI consumption."""

    return list(UNLEARNING_STRATEGIES)


def get_strategy(strategy_id: str) -> Optional[UnlearningStrategy]:
    for strategy in UNLEARNING_STRATEGIES:
        if strategy.id == strategy_id:
            return strategy
    return None


def build_unlearning_prompt(
    strategy_id: str,
    target_description: str,
    *,
    custom_prompt: Optional[str] = None,
) -> str:
    """Construct a prompt for unlearning detection."""

    if custom_prompt and strategy_id == "custom":
        template = custom_prompt
    else:
        strategy = get_strategy(strategy_id)
        if strategy is None:
            raise ValueError(f"Unknown strategy id: {strategy_id}")
        template = strategy.template

    target = target_description.strip() or "the withheld passage"
    return template.format(target_description=target)

def _chunk_text_by_words(text: str, *, chunk_size: int, max_chunks: Optional[int] = None) -> List[str]:
    """Split text into word-based chunks suitable for perplexity evaluation."""

    words = text.split()
    if not words or chunk_size <= 0:
        return []

    chunks: List[str] = []
    total = len(words)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        snippet = " ".join(chunk_words).strip()
        if not snippet:
            continue
        chunks.append(snippet)
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def _chunk_text_for_membership(
    text: str,
    *,
    chunk_size: int,
    max_chunks: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> Tuple[List[str], Optional[str]]:
    """Return token-based chunks when a tokenizer is available, otherwise fall back to words."""

    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception:  # pragma: no cover - encoder errors
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        if not tokens or chunk_size <= 0:
            return [], None

        chunks: List[str] = []
        total = len(tokens)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            token_slice = tokens[start:end]
            if not token_slice:
                continue
            decoded = encoding.decode(token_slice).strip()
            if not decoded:
                continue
            chunks.append(decoded)
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
        return chunks, None

    # Fallback to word chunking when tokenizer isn't installed
    return _chunk_text_by_words(text, chunk_size=chunk_size, max_chunks=max_chunks), (
        "Tokeniser 'tiktoken' not installed—falling back to word-based chunks."
    )


def _fetch_logprobs_for_text(
    api_key: str,
    model_name: str,
    provider: str,
    text: str,
) -> Tuple[List[float], Optional[str]]:
    """Return log probabilities for each prompt token."""

    if provider != "OpenAI":
        return [], f"Logprob extraction is currently supported for OpenAI models only (got {provider})."

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.completions.create(
            model=model_name,
            prompt=text,
            max_tokens=0,
            temperature=0.0,
            logprobs=1,
            echo=True,
        )
    except Exception as exc:  # pragma: no cover - network/SDK errors
        fallback_reason = str(exc)
        is_bad_request = BadRequestError is not None and isinstance(exc, BadRequestError)
        invalid_combo = "invalid_parameter_combination" in fallback_reason or "Setting 'echo' and 'logprobs'" in fallback_reason

        if is_bad_request and not invalid_combo:
            return [], f"Failed to retrieve logprobs: {exc}"

        values, error = _fetch_logprobs_via_responses(client, model_name, text)
        if error:
            return [], f"Failed to retrieve logprobs: {fallback_reason}; {error}"
        return values, None

    choice = response.choices[0]
    logprobs = choice.logprobs
    if not logprobs or not logprobs.token_logprobs:
        # Attempt to fall back to the Responses API when prompt logprobs aren't returned.
        values, error = _fetch_logprobs_via_responses(client, model_name, text)
        if error:
            return [], error
        return values, None

    values = [lp for lp in logprobs.token_logprobs if lp is not None]
    if not values:
        return [], "Token log probabilities were empty."

    return values, None


def _compute_perplexity(logprobs: Sequence[float]) -> Tuple[float, float]:
    if not logprobs:
        return float("nan"), float("inf")

    avg_logprob = fmean(logprobs)
    perplexity = math.exp(-avg_logprob)
    return avg_logprob, perplexity


def _normalise_openai_obj(value):  # type: ignore[override]
    """Return dict-like access for OpenAI objects or plain dicts."""

    if value is None:
        return None
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if isinstance(value, dict):
        return value
    return value


def _extract_token_logprobs_from_prompt_section(prompt_section) -> List[float]:
    values: List[float] = []
    if not prompt_section:
        return values

    section_iterable = prompt_section if isinstance(prompt_section, (list, tuple)) else [prompt_section]
    for segment in section_iterable:
        segment_dict = _normalise_openai_obj(segment)
        if segment_dict is None:
            continue
        content_items = segment_dict.get("content") if isinstance(segment_dict, dict) else getattr(segment, "content", None)
        if not content_items:
            continue
        for item in content_items:
            item_dict = _normalise_openai_obj(item)
            if item_dict is None:
                continue
            logprob_block = item_dict.get("logprobs") if isinstance(item_dict, dict) else getattr(item, "logprobs", None)
            if not logprob_block:
                continue
            block_dict = _normalise_openai_obj(logprob_block)
            if block_dict is None:
                continue
            token_logprobs = (
                block_dict.get("token_logprobs")
                if isinstance(block_dict, dict)
                else getattr(logprob_block, "token_logprobs", None)
            )
            if not token_logprobs:
                continue
            values.extend(lp for lp in token_logprobs if lp is not None)
    return values


def _fetch_logprobs_via_responses(
    client: "openai.OpenAI",
    model_name: str,
    text: str,
) -> Tuple[List[float], Optional[str]]:
    """Fallback path using the Responses API for models that disallow echo logprobs."""

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            ],
            logprobs=True,
            max_output_tokens=0,
            temperature=0.0,
        )
    except Exception as exc:  # pragma: no cover - network/SDK errors
        return [], f"Responses API fallback failed: {exc}"

    prompt_section = getattr(response, "prompt", None) or getattr(response, "input", None)
    values = _extract_token_logprobs_from_prompt_section(prompt_section)
    if values:
        return values, None

    # Some SDK versions expose prompt logprobs under response.metadata["prompt"]
    metadata = getattr(response, "metadata", None)
    if metadata and isinstance(metadata, dict):
        prompt_meta = metadata.get("prompt")
        values = _extract_token_logprobs_from_prompt_section(prompt_meta)
        if values:
            return values, None

    return [], (
        "Provider returned no token log probabilities via responses API. "
        "Ensure the selected model supports logprob outputs for prompt tokens."
    )


def _safe_median(values: Sequence[float]) -> float:
    if not values:
        return float("inf")
    try:
        return median(values)
    except StatisticsError:  # pragma: no cover - safeguard against degenerate input
        return float("inf")


def _safe_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    try:
        return stdev(values)
    except StatisticsError:  # pragma: no cover - safeguard against degenerate input
        return float("nan")


def _hedges_g(control: Sequence[float], target: Sequence[float]) -> Optional[float]:
    if len(control) < 2 or len(target) < 2:
        return None

    mean_control = fmean(control)
    mean_target = fmean(target)
    var_control = variance(control)
    var_target = variance(target)

    n_control = len(control)
    n_target = len(target)
    pooled_denominator = (n_control + n_target - 2)
    if pooled_denominator <= 0:
        return None

    pooled_variance = ((n_control - 1) * var_control + (n_target - 1) * var_target) / pooled_denominator
    if pooled_variance <= 0:
        return None

    d = (mean_control - mean_target) / math.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (n_control + n_target) - 9)
    return d * correction


def _bootstrap_gap_confidence_interval(
    target: Sequence[float],
    control: Sequence[float],
    *,
    samples: int,
    confidence: float,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[float, float]]:
    if samples <= 0 or len(target) < 2 or len(control) < 2:
        return None

    rng = rng or random.Random()
    deltas: List[float] = []
    lower_tail = (1 - confidence) / 2
    upper_tail = 1 - lower_tail

    for _ in range(samples):
        resampled_target = [rng.choice(target) for _ in range(len(target))]
        resampled_control = [rng.choice(control) for _ in range(len(control))]
        mean_target = fmean(resampled_target)
        mean_control = fmean(resampled_control)
        deltas.append(mean_control - mean_target)

    if not deltas:
        return None

    deltas.sort()
    lower_index = max(0, min(samples - 1, int(lower_tail * samples)))
    upper_index = max(0, min(samples - 1, int(math.ceil(upper_tail * samples)) - 1))

    return deltas[lower_index], deltas[upper_index]


def _annotate_relative_metrics(
    *,
    target_results: Sequence[MembershipSnippetResult],
    control_results: Sequence[MembershipSnippetResult],
    mean_control: float,
    std_control: float,
) -> None:
    if math.isinf(mean_control):
        return

    has_std = not math.isnan(std_control) and std_control not in (0.0, float("inf"))
    combined = list(target_results) + list(control_results)
    for result in combined:
        if result.error or math.isinf(result.perplexity):
            result.relative_perplexity = None
            result.z_score = None
            continue

        result.relative_perplexity = mean_control - result.perplexity
        if has_std:
            result.z_score = (mean_control - result.perplexity) / std_control
        else:
            result.z_score = None


def _summarize_membership_results(
    target_results: Sequence[MembershipSnippetResult],
    control_results: Sequence[MembershipSnippetResult],
    *,
    ppl_gap_threshold: float,
    additional_errors: Sequence[str],
    bootstrap_samples: int,
    confidence_level: float,
    rng: Optional[random.Random] = None,
) -> MembershipInferenceSummary:
    valid_target = [res.perplexity for res in target_results if not math.isinf(res.perplexity) and not math.isnan(res.perplexity)]
    valid_control = [res.perplexity for res in control_results if not math.isinf(res.perplexity) and not math.isnan(res.perplexity)]

    mean_target = fmean(valid_target) if valid_target else float("inf")
    mean_control = fmean(valid_control) if valid_control else float("inf")
    median_target = _safe_median(valid_target)
    median_control = _safe_median(valid_control)
    std_target = _safe_stdev(valid_target)
    std_control = _safe_stdev(valid_control)

    ppl_gap = (
        mean_control - mean_target
        if valid_target and valid_control and not math.isinf(mean_target) and not math.isinf(mean_control)
        else float("nan")
    )
    flagged = bool(valid_target and valid_control and not math.isnan(ppl_gap) and ppl_gap >= ppl_gap_threshold)

    tests: List[StatisticalTestOutcome] = []
    if valid_target and valid_control:
        tests.append(_welch_t_test(valid_target, valid_control))
        tests.append(_ks_test(valid_target, valid_control))

    effect_size = _hedges_g(valid_control, valid_target)
    ppl_gap_ci = _bootstrap_gap_confidence_interval(
        valid_target,
        valid_control,
        samples=bootstrap_samples,
        confidence=confidence_level,
        rng=rng,
    ) if bootstrap_samples > 0 else None

    _annotate_relative_metrics(
        target_results=target_results,
        control_results=control_results,
        mean_control=mean_control,
        std_control=std_control,
    )

    combined_errors: List[str] = list(additional_errors)
    if not valid_target:
        combined_errors.append("No valid reference perplexity samples were collected.")
    if not valid_control:
        combined_errors.append("No valid control perplexity samples were collected.")

    errors = list(dict.fromkeys(err for err in combined_errors if err))

    return MembershipInferenceSummary(
        target_results=list(target_results),
        control_results=list(control_results),
        mean_target_ppl=mean_target,
        mean_control_ppl=mean_control,
        ppl_gap=ppl_gap,
        threshold=ppl_gap_threshold,
        flagged=flagged,
        sample_sizes=(len(valid_target), len(valid_control)),
        statistical_tests=tests,
        errors=errors,
        median_target_ppl=median_target,
        median_control_ppl=median_control,
        std_target_ppl=std_target,
        std_control_ppl=std_control,
        effect_size=effect_size,
        ppl_gap_ci=ppl_gap_ci,
        bootstrap_iterations=bootstrap_samples if bootstrap_samples > 0 else 0,
    )


@dataclass
class MembershipSnippetResult:
    label: str
    snippet: str
    token_count: int
    avg_logprob: float
    perplexity: float
    logprobs: List[float]
    error: Optional[str] = None
    relative_perplexity: Optional[float] = None
    z_score: Optional[float] = None

    @property
    def training_trace_score(self) -> float:
        if self.perplexity in (0, float("inf")):
            return 0.0
        return 1.0 / self.perplexity


@dataclass
class MembershipInferenceSummary:
    target_results: List[MembershipSnippetResult]
    control_results: List[MembershipSnippetResult]
    mean_target_ppl: float
    mean_control_ppl: float
    ppl_gap: float
    threshold: float
    flagged: bool
    sample_sizes: Tuple[int, int]
    statistical_tests: List[StatisticalTestOutcome]
    errors: List[str]
    median_target_ppl: float = float("inf")
    median_control_ppl: float = float("inf")
    std_target_ppl: float = float("nan")
    std_control_ppl: float = float("nan")
    effect_size: Optional[float] = None
    ppl_gap_ci: Optional[Tuple[float, float]] = None
    bootstrap_iterations: int = 0

    @property
    def training_signal_score(self) -> float:
        if not self.control_results:
            return 0.0
        denominator = max(self.mean_control_ppl, 1e-6)
        raw = max(0.0, self.mean_control_ppl - self.mean_target_ppl) / denominator
        return min(raw, 1.0)


def _welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> StatisticalTestOutcome:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return StatisticalTestOutcome(
            name="Welch's t-test",
            statistic=None,
            pvalue=None,
            detail="Need at least two valid samples per group to run Welch's t-test.",
        )

    if scipy_stats is not None:  # pragma: no cover - requires scipy
        result = scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False)
        stat = float(result.statistic) if result.statistic is not None else None
        pvalue = float(result.pvalue) if result.pvalue is not None else None
        return StatisticalTestOutcome(
            name="Welch's t-test",
            statistic=stat,
            pvalue=pvalue,
        )

    mean_a = fmean(sample_a)
    mean_b = fmean(sample_b)
    var_a = variance(sample_a)
    var_b = variance(sample_b)
    n_a = len(sample_a)
    n_b = len(sample_b)

    se = math.sqrt((var_a / n_a) + (var_b / n_b))
    if se == 0:
        return StatisticalTestOutcome(
            name="Welch's t-test",
            statistic=None,
            pvalue=None,
            detail="Unable to compute t-statistic because the standard error is zero.",
        )

    t_stat = (mean_a - mean_b) / se

    # Welch–Satterthwaite degrees of freedom
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    dof = numerator / denominator if denominator else float("inf")

    # Gaussian approximation for p-value when SciPy isn't available
    z = abs(t_stat)
    pvalue = math.erfc(z / math.sqrt(2))

    detail = (
        "SciPy not available—computed Welch's t-statistic with a Gaussian-tail approximation for the p-value "
        f"(dof≈{dof:.1f})."
    )
    return StatisticalTestOutcome(
        name="Welch's t-test",
        statistic=t_stat,
        pvalue=pvalue,
        detail=detail,
    )


def _ks_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> StatisticalTestOutcome:
    if not sample_a or not sample_b:
        return StatisticalTestOutcome(
            name="Kolmogorov–Smirnov",
            statistic=None,
            pvalue=None,
            detail="Need at least one sample in each group to run the KS test.",
        )

    if scipy_stats is not None:  # pragma: no cover - requires scipy
        result = scipy_stats.ks_2samp(sample_a, sample_b)
        return StatisticalTestOutcome(
            name="Kolmogorov–Smirnov",
            statistic=float(result.statistic),
            pvalue=float(result.pvalue),
        )

    sorted_a = sorted(sample_a)
    sorted_b = sorted(sample_b)
    n = len(sorted_a)
    m = len(sorted_b)

    i = j = 0
    cdf_a = cdf_b = 0.0
    d_max = 0.0
    while i < n and j < m:
        if sorted_a[i] <= sorted_b[j]:
            cdf_a = (i + 1) / n
            i += 1
        else:
            cdf_b = (j + 1) / m
            j += 1
        d_max = max(d_max, abs(cdf_a - cdf_b))

    d_max = max(d_max, abs(1 - cdf_b), abs(1 - cdf_a))

    en = math.sqrt(n * m / (n + m))
    if en == 0:
        return StatisticalTestOutcome(
            name="Kolmogorov–Smirnov",
            statistic=None,
            pvalue=None,
            detail="Unable to estimate KS statistic due to zero effective sample size.",
        )

    lam = (en + 0.12 + 0.11 / en) * d_max
    # Kolmogorov cumulative distribution approximation (two-sided)
    # Q_KS(lambda) = 2 * sum_{j=1}^∞ (-1)^{j-1} exp(-2 j^2 lambda^2)
    pvalue = 0.0
    for j in range(1, 6):
        term = (-1) ** (j - 1) * math.exp(-2 * (lam ** 2) * (j ** 2))
        pvalue += term
    pvalue = max(0.0, min(1.0, 2 * pvalue))

    detail = "SciPy not available—approximated KS p-value via asymptotic expansion."
    return StatisticalTestOutcome(
        name="Kolmogorov–Smirnov",
        statistic=d_max,
        pvalue=pvalue,
        detail=detail,
    )


def run_membership_inference(
    api_key: str,
    model_name: str,
    provider: str,
    *,
    reference_text: str,
    control_text: str,
    chunk_size: int = 120,
    max_chunks: int = 4,
    ppl_gap_threshold: float = 5.0,
    bootstrap_samples: int = 500,
    confidence_level: float = 0.90,
    rng: Optional[random.Random] = None,
) -> MembershipInferenceSummary:
    """Compute perplexity-based membership inference statistics."""

    reference_chunks, reference_warning = _chunk_text_for_membership(
        reference_text,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )
    control_chunks, control_warning = _chunk_text_for_membership(
        control_text,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )

    if not reference_chunks:
        raise ValueError("Reference text is too short to generate membership probes.")
    if not control_chunks:
        raise ValueError("Control text is required to establish a perplexity baseline.")

    errors: List[str] = []
    for warning in (reference_warning, control_warning):
        if warning:
            errors.append(warning)
    target_results: List[MembershipSnippetResult] = []
    control_results: List[MembershipSnippetResult] = []

    def evaluate_chunks(chunks: Sequence[str], label: str) -> List[MembershipSnippetResult]:
        evaluations: List[MembershipSnippetResult] = []
        for chunk in chunks:
            logprobs, err = _fetch_logprobs_for_text(api_key, model_name, provider, chunk)
            if err:
                errors.append(err)
                evaluations.append(
                    MembershipSnippetResult(
                        label=label,
                        snippet=chunk,
                        token_count=0,
                        avg_logprob=float("nan"),
                        perplexity=float("inf"),
                        logprobs=[],
                        error=err,
                    )
                )
                continue

            avg_logprob, perplexity = _compute_perplexity(logprobs)
            evaluations.append(
                MembershipSnippetResult(
                    label=label,
                    snippet=chunk,
                    token_count=len(logprobs),
                    avg_logprob=avg_logprob,
                    perplexity=perplexity,
                    logprobs=list(logprobs),
                )
            )
        return evaluations

    target_results = evaluate_chunks(reference_chunks, "reference")
    control_results = evaluate_chunks(control_chunks, "control")

    summary = _summarize_membership_results(
        target_results=target_results,
        control_results=control_results,
        ppl_gap_threshold=ppl_gap_threshold,
        additional_errors=errors,
        bootstrap_samples=max(0, int(bootstrap_samples)),
        confidence_level=max(0.0, min(0.999, confidence_level)),
        rng=rng,
    )

    return summary


@dataclass
class UnlearningProbeResult:
    strategy_id: str
    strategy_name: str
    prompt: str
    response: str
    error: Optional[str] = None


@dataclass
class UnlearningDetectionSummary:
    results: List[UnlearningProbeResult]


def run_unlearning_detection(
    api_key: str,
    model_name: str,
    provider: str,
    *,
    target_description: str,
    strategy_ids: Iterable[str],
    temperature: float = 0.7,
    top_p: float = 1.0,
    custom_prompt: Optional[str] = None,
) -> UnlearningDetectionSummary:
    """Execute detection probes and return aggregated results."""

    strategies: List[str] = list(strategy_ids)
    if not strategies:
        raise ValueError("At least one strategy must be selected")

    results: List[UnlearningProbeResult] = []

    for strategy_id in strategies:
        prompt = build_unlearning_prompt(strategy_id, target_description, custom_prompt=custom_prompt)
        response = get_llm_completion(
            prompt,
            api_key,
            model_name,
            provider,
            temperature=temperature,
            top_p=top_p,
        )

        if isinstance(response, str) and response.startswith("Error"):
            results.append(
                UnlearningProbeResult(
                    strategy_id=strategy_id,
                    strategy_name=(get_strategy(strategy_id).name if get_strategy(strategy_id) else "Custom"),
                    prompt=prompt,
                    response=response,
                    error=response,
                )
            )
            continue

        results.append(
            UnlearningProbeResult(
                strategy_id=strategy_id,
                strategy_name=(get_strategy(strategy_id).name if get_strategy(strategy_id) else "Custom"),
                prompt=prompt,
                response=response,
            )
        )

    summary = UnlearningDetectionSummary(results=results)
    return summary


__all__ = [
    "UnlearningStrategy",
    "UnlearningProbeResult",
    "UnlearningDetectionSummary",
    "MembershipSnippetResult",
    "MembershipInferenceSummary",
    "StatisticalTestOutcome",
    "UNLEARNING_STRATEGIES",
    "RepresentationalFeature",
    "RepresentationalAnalysisResult",
    "REPRESENTATIONAL_FEATURES",
    "build_unlearning_prompt",
    "list_unlearning_strategies",
    "list_representational_features",
    "get_representational_feature",
    "is_representational_analysis_available",
    "run_unlearning_detection",
    "run_membership_inference",
    "run_representational_analysis",
]
