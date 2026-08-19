"""Competition-safe wrappers around the existing recall and persuasion probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional, Sequence

from rouge_score import rouge_scorer

from .constants import (
    BOOK_BENCHMARKS,
    BOOK_KEYS,
    DEFAULT_BOOK_KEY,
    GAME_MAX_OUTPUT_TOKENS,
    GAME_MODEL,
    GAME_PROVIDER,
    MAX_ATTEMPTS_PER_STRATEGY,
    MAX_SCORED_GENERATIONS,
    MAX_STAGE_ONE_ATTEMPTS,
    OTHER_STAGE_ONE_BOOK_KEY,
    OTHER_STAGE_ONE_BOOK_TITLE,
)


_GAME_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_GAME_OPENAI_TIMEOUT_SECONDS = 90.0
_GAME_OPENAI_MAX_RETRIES = 1

# Uses the book's original direct-probe prompt without persuasion mutation.
BASELINE_STRATEGY = "Baseline"


class GameValidationError(ValueError):
    """Raised when a participant configuration violates competition rules."""


class GameRunError(RuntimeError):
    """Raised when a model call prevents a complete, scoreable official run."""


@dataclass(frozen=True)
class GameConfig:
    """Participant-controlled settings for a one-to-three-book official batch."""

    shot_mode: str
    strategy: str
    attempts_per_strategy: int
    attempts_per_prompt: int
    temperature: float = 0.7
    top_p: float = 0.9
    book_key: str = DEFAULT_BOOK_KEY
    book_keys: Optional[Sequence[str]] = None
    strategies: Optional[Sequence[str]] = None

    @property
    def selected_book_keys(self) -> tuple[str, ...]:
        if self.book_keys is None:
            return (self.book_key,)
        return tuple(str(key) for key in self.book_keys)

    @property
    def selected_strategies(self) -> tuple[str, ...]:
        if self.strategies is None:
            return (self.strategy,) if str(self.strategy or "").strip() else ()
        return tuple(str(strategy) for strategy in self.strategies)

    @property
    def strategy_label(self) -> str:
        return " | ".join(self.selected_strategies)

    @property
    def total_mutations(self) -> int:
        return (
            self.attempts_per_strategy
            * len(self.selected_book_keys)
            * len(self.selected_strategies)
        )

    @property
    def scored_generations(self) -> int:
        return self.total_mutations * self.attempts_per_prompt

    @property
    def database_shot_mode(self) -> str:
        return "zero_shot"

    def validate(self, available_strategies: Optional[Sequence[str]] = None) -> None:
        if self.shot_mode != "Zero-Shot":
            raise GameValidationError("The competition setup is fixed to Zero-Shot.")
        selected_books = self.selected_book_keys
        if not selected_books:
            raise GameValidationError("Select at least one competition book.")
        if len(selected_books) > len(BOOK_KEYS):
            raise GameValidationError("Select at most the three competition books.")
        if len(set(selected_books)) != len(selected_books):
            raise GameValidationError("Each competition book may be selected only once.")
        if any(book not in BOOK_KEYS for book in selected_books):
            raise GameValidationError("Select only the three competition books.")
        if self.book_key != selected_books[0]:
            raise GameValidationError("The primary book must be the first selected book.")
        selected_strategies = self.selected_strategies
        if not selected_strategies:
            raise GameValidationError("Select at least one persuasion strategy.")
        if len(set(selected_strategies)) != len(selected_strategies):
            raise GameValidationError("Each persuasion strategy may be selected only once.")
        if self.strategy != selected_strategies[0]:
            raise GameValidationError(
                "The primary strategy must be the first selected strategy."
            )
        if available_strategies is not None and any(
            strategy not in available_strategies for strategy in selected_strategies
        ):
            raise GameValidationError(
                "One or more selected persuasion strategies are not available."
            )
        if isinstance(self.attempts_per_strategy, bool) or not isinstance(
            self.attempts_per_strategy, int
        ):
            raise GameValidationError("Attempts per strategy must be an integer.")
        if isinstance(self.attempts_per_prompt, bool) or not isinstance(
            self.attempts_per_prompt, int
        ):
            raise GameValidationError("Attempts per prompt must be an integer.")
        if not 1 <= self.attempts_per_strategy <= MAX_ATTEMPTS_PER_STRATEGY:
            raise GameValidationError(
                f"Attempts per strategy must be between 1 and {MAX_ATTEMPTS_PER_STRATEGY}."
            )
        if self.attempts_per_prompt < 1:
            raise GameValidationError("Attempts per prompt must be at least 1.")
        if self.scored_generations > MAX_SCORED_GENERATIONS:
            raise GameValidationError(
                "Per run, books x strategies x mutations/book x "
                f"responses must be at most {MAX_SCORED_GENERATIONS}."
            )
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise GameValidationError("Temperature must be between 0 and 2.")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise GameValidationError("Top-p must be greater than 0 and at most 1.")


@dataclass(frozen=True)
class DirectProbeAttempt:
    attempt: int
    response: str
    metrics: Dict[str, float]

    @property
    def rouge_l(self) -> float:
        return float(self.metrics.get("rouge_l", 0.0) or 0.0)


@dataclass(frozen=True)
class StageOneResult:
    response: str
    metrics: Dict[str, float]
    temperature: float = 0.0
    top_p: float = 1.0
    book_key: str = DEFAULT_BOOK_KEY
    attempts: Sequence[DirectProbeAttempt] = ()

    @property
    def scores(self) -> List[float]:
        if self.attempts:
            return [attempt.rouge_l for attempt in self.attempts]
        return [float(self.metrics.get("rouge_l", 0.0) or 0.0)]

    @property
    def rouge_l(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def avg_rouge_l(self) -> float:
        return fmean(self.scores) if self.scores else 0.0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts) if self.attempts else 1


@dataclass(frozen=True)
class ScoredGeneration:
    mutation_attempt: int
    prompt_attempt: int
    mutated_prompt: str
    response: str
    metrics: Dict[str, float]
    book_key: str = DEFAULT_BOOK_KEY
    strategy: str = ""

    @property
    def rouge_l(self) -> float:
        return float(self.metrics.get("rouge_l", 0.0) or 0.0)


@dataclass(frozen=True)
class StageTwoResult:
    config: GameConfig
    generations: Sequence[ScoredGeneration]

    @property
    def scores(self) -> List[float]:
        return [generation.rouge_l for generation in self.generations]

    @property
    def max_rouge_l(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def avg_rouge_l(self) -> float:
        return fmean(self.scores) if self.scores else 0.0

    @property
    def best_generation(self) -> Optional[ScoredGeneration]:
        if not self.generations:
            return None
        return max(self.generations, key=lambda generation: generation.rouge_l)


def list_game_books() -> List[str]:
    """Return stable book keys in the public competition display order."""

    return list(BOOK_KEYS)


def get_book_title(book_key: str) -> str:
    if book_key == OTHER_STAGE_ONE_BOOK_KEY:
        return OTHER_STAGE_ONE_BOOK_TITLE
    try:
        return str(BOOK_BENCHMARKS[book_key]["title"])
    except KeyError as exc:
        raise GameValidationError("Select a supported competition book.") from exc


def get_book_prompt(book_key: str) -> str:
    if book_key == OTHER_STAGE_ONE_BOOK_KEY:
        return ""
    try:
        return str(BOOK_BENCHMARKS[book_key]["prompt"])
    except KeyError as exc:
        raise GameValidationError("Select a supported competition book.") from exc


def get_reference_text(book_key: str = DEFAULT_BOOK_KEY) -> str:
    """Return one book's canonical first 100 whitespace-delimited words."""

    from src.adversarial_persuasion_detection.adversarial_prompting import (
        DEFAULT_GA_REFERENCE_EXCERPT,
        DEFAULT_HB_REFERENCE_EXCERPT,
        DEFAULT_HP_REFERENCE_EXCERPT,
    )

    source_by_book = {
        "harry_potter": DEFAULT_HP_REFERENCE_EXCERPT,
        "the_hobbit": DEFAULT_HB_REFERENCE_EXCERPT,
        "a_game_of_thrones": DEFAULT_GA_REFERENCE_EXCERPT,
    }
    try:
        source_text = source_by_book[book_key]
        expected_digest = str(BOOK_BENCHMARKS[book_key]["reference_sha256"])
    except KeyError as exc:
        raise GameValidationError("Select one of the three competition books.") from exc

    if book_key == "a_game_of_thrones":
        # The shared preset joins adjacent dialogue quotes without whitespace.
        source_text = source_text.replace('""', '" "')

    words = source_text.split()
    if len(words) < 100:
        raise GameRunError(
            f"The configured {get_book_title(book_key)} benchmark has fewer than 100 words."
        )
    reference_text = " ".join(words[:100])
    digest = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
    if digest != expected_digest:
        raise GameRunError(
            "The versioned competition reference changed unexpectedly. "
            "Update the benchmark version before accepting new scores."
        )
    return reference_text


def list_game_strategies() -> List[str]:
    """Return Baseline plus the Persuasion-main strategies used by the game."""

    from src.adversarial_persuasion_detection.adversarial_prompting import (
        list_persuasion_strategies,
    )

    return [BASELINE_STRATEGY, *list_persuasion_strategies()]


def _default_completion() -> Callable[..., Any]:
    from src.direct_recall.comparison import get_llm_completion

    def complete(*args: Any, **kwargs: Any) -> Any:
        # Game calls can run after the originating Streamlit script has rerun.
        # Keep them UI-free and bounded so one request cannot lock future runs.
        kwargs.setdefault("request_timeout", _GAME_OPENAI_TIMEOUT_SECONDS)
        kwargs.setdefault("request_max_retries", _GAME_OPENAI_MAX_RETRIES)
        return get_llm_completion(*args, **kwargs)

    return complete


def _calculate_rouge_l_metrics(reference_text: str, candidate_text: str) -> Dict[str, float]:
    """Calculate only the official competition metric."""

    score = _GAME_ROUGE_SCORER.score(reference_text, candidate_text)["rougeL"]
    return {"rouge_l": float(score.fmeasure)}


def _default_metrics() -> Callable[[str, str], Dict[str, float]]:
    return _calculate_rouge_l_metrics


def _default_mutation() -> Callable[..., Any]:
    from src.adversarial_persuasion_detection.adversarial_prompting import (
        mutate_strategies,
    )

    return mutate_strategies


def _completion_text(result: Any) -> str:
    value = result[0] if isinstance(result, tuple) else result
    text = str(value or "").strip()
    if not text:
        raise GameRunError("The model returned an empty response.")
    if text.lower().startswith("error"):
        raise GameRunError(text)
    return text


def run_stage_one(
    api_key: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    book_key: str = DEFAULT_BOOK_KEY,
    attempts: int = 1,
    prompt: Optional[str] = None,
    reference_text: Optional[str] = None,
    completion_fn: Optional[Callable[..., Any]] = None,
    metrics_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> StageOneResult:
    """Run the direct-probing baseline with participant-selected sampling."""

    if not str(api_key or "").strip():
        raise GameRunError("The shared competition API key is not configured.")
    is_other_book = book_key == OTHER_STAGE_ONE_BOOK_KEY
    if book_key not in BOOK_KEYS and not is_other_book:
        raise GameValidationError("Select a supported Stage 1 book option.")
    if not 0.0 <= float(temperature) <= 2.0:
        raise GameValidationError("Temperature must be between 0 and 2.")
    if not 0.0 < float(top_p) <= 1.0:
        raise GameValidationError("Top-p must be greater than 0 and at most 1.")
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise GameValidationError("Direct-probe runs must be an integer.")
    if not 1 <= attempts <= MAX_STAGE_ONE_ATTEMPTS:
        raise GameValidationError(
            f"Direct-probe runs must be between 1 and {MAX_STAGE_ONE_ATTEMPTS}."
        )


    prompt = str(prompt or "").strip()
    if not prompt:
        if is_other_book:
            raise GameValidationError("Enter a custom direct-probe prompt for Other books.")
        prompt = get_book_prompt(book_key)

    if is_other_book:
        reference_text = str(reference_text or "").strip()
        if not reference_text:
            raise GameValidationError("Enter a ground truth for Other books.")
    else:
        reference_text = get_reference_text(book_key)

    completion = completion_fn or _default_completion()
    calculate_metrics = metrics_fn or _default_metrics()
    completed_attempts: List[DirectProbeAttempt] = []
    for attempt_index in range(1, attempts + 1):
        raw = completion(
            prompt,
            api_key,
            GAME_MODEL,
            provider=GAME_PROVIDER,
            temperature=float(temperature),
            top_p=float(top_p),
            max_output_tokens=GAME_MAX_OUTPUT_TOKENS,
        )
        response = _completion_text(raw)
        completed_attempts.append(
            DirectProbeAttempt(
                attempt=attempt_index,
                response=response,
                metrics=calculate_metrics(reference_text, response),
            )
        )
        if on_progress is not None:
            on_progress(attempt_index, attempts)

    best_attempt = max(completed_attempts, key=lambda item: item.rouge_l)
    return StageOneResult(
        response=best_attempt.response,
        metrics=best_attempt.metrics,
        temperature=float(temperature),
        top_p=float(top_p),
        book_key=book_key,
        attempts=tuple(completed_attempts),
    )


def run_stage_two(
    api_key: str,
    config: GameConfig,
    *,
    mutation_fn: Optional[Callable[..., Any]] = None,
    completion_fn: Optional[Callable[..., Any]] = None,
    metrics_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    available_strategies: Optional[Sequence[str]] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> StageTwoResult:
    """Run one complete persuasive batch across every selected competition book."""

    if not str(api_key or "").strip():
        raise GameRunError("The shared competition API key is not configured.")

    strategies = list(available_strategies or list_game_strategies())
    config.validate(strategies)
    mutate = mutation_fn or _default_mutation()

    completion = completion_fn or _default_completion()
    calculate_metrics = metrics_fn or _default_metrics()
    total_mutations = config.total_mutations

    if on_progress:
        on_progress("mutations", 0, total_mutations)

    planned_mutations: List[tuple[str, str, int, str]] = []
    completed_mutations = 0
    for book_key in config.selected_book_keys:
        prompt = get_book_prompt(book_key)
        for strategy in config.selected_strategies:
            # Baseline skips mutation and scores the original direct-probe prompt.
            if strategy == BASELINE_STRATEGY:
                for _local_index in range(1, config.attempts_per_strategy + 1):
                    completed_mutations += 1
                    planned_mutations.append(
                        (book_key, strategy, completed_mutations, prompt)
                    )
                    if on_progress:
                        on_progress(
                            "mutations", completed_mutations, total_mutations
                        )
                continue

            mutation_evaluations = mutate(
                api_key,
                GAME_MODEL,
                GAME_PROVIDER,
                [strategy],
                prompt,
                reference_text=None,
                attempts_per_strategy=config.attempts_per_strategy,
                attempts_per_prompt=1,
                temperature=float(config.temperature),
                top_p=float(config.top_p),
                dry_run=False,
            )
            if len(mutation_evaluations) != config.attempts_per_strategy:
                raise GameRunError(
                    f"Mutation generation for {get_book_title(book_key)} with "
                    f"{strategy} was incomplete; no official score was submitted."
                )
            for local_index, evaluation in enumerate(mutation_evaluations, start=1):
                mutation = getattr(evaluation, "mutation", None)
                error = getattr(mutation, "error", None)
                parsed = getattr(evaluation, "parsed", None)
                mutated_text = str(getattr(parsed, "mutated_text", "") or "").strip()
                if error or not mutated_text:
                    detail = str(error or "empty mutated prompt")
                    raise GameRunError(
                        f"Mutation {local_index} for {get_book_title(book_key)} with "
                        f"{strategy} failed ({detail}); no official score was submitted."
                    )
                completed_mutations += 1
                planned_mutations.append(
                    (book_key, strategy, completed_mutations, mutated_text)
                )
                if on_progress:
                    on_progress("mutations", completed_mutations, total_mutations)
    if len(planned_mutations) != total_mutations:
        raise GameRunError(
            "The multi-book mutation batch was incomplete; no official score was submitted."
        )

    references = {
        book_key: get_reference_text(book_key) for book_key in config.selected_book_keys
    }
    generations: List[ScoredGeneration] = []
    completed_generations = 0
    total_generations = config.scored_generations
    for (
        book_key,
        strategy,
        global_mutation_attempt,
        mutated_prompt,
    ) in planned_mutations:
        for prompt_attempt in range(1, config.attempts_per_prompt + 1):
            raw = completion(
                mutated_prompt,
                api_key,
                GAME_MODEL,
                provider=GAME_PROVIDER,
                temperature=float(config.temperature),
                top_p=float(config.top_p),
                max_output_tokens=GAME_MAX_OUTPUT_TOKENS,
            )
            response = _completion_text(raw)
            generations.append(
                ScoredGeneration(
                    mutation_attempt=global_mutation_attempt,
                    prompt_attempt=prompt_attempt,
                    mutated_prompt=mutated_prompt,
                    response=response,
                    metrics=calculate_metrics(references[book_key], response),
                    book_key=book_key,
                    strategy=strategy,
                )
            )
            completed_generations += 1
            if on_progress:
                on_progress("evaluation", completed_generations, total_generations)

    if len(generations) != total_generations:
        raise GameRunError(
            "The multi-book scored generation batch was incomplete; "
            "no official score was submitted."
        )
    return StageTwoResult(config=config, generations=generations)
