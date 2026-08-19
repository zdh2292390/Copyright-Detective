"""Competition-safe Knowledge Memorization engine for Copyright Challenge 3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.direct_recall.knowledge_qa import (
    answer_question_with_llm,
    evaluate_qa_comparison,
)
from src.direct_recall.sleek_attack import decompose_question, run_cot_reasoning

from .constants import (
    BOOK_KEYS,
    DEFAULT_BOOK_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GAME_MODEL,
    GAME_PROVIDER,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_RUNS,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    PROBE_MODES,
    QUESTIONS_PER_BOOK,
    SLEEK_MODE,
    STANDARD_MODE,
    get_book_title,
    get_question_bank,
    list_game_books,
    max_runs_for_mode,
)


class GameValidationError(ValueError):
    """Raised when a Game 3 configuration violates the frozen rules."""


class GameRunError(RuntimeError):
    """Raised when a model call prevents a complete, scoreable run."""


@dataclass(frozen=True)
class KnowledgeGameConfig:
    """Participant-controlled settings dedicated to Knowledge Memorization.

    A run always evaluates all five questions for one book. ``repetitions`` is
    capped by mode: Standard allows 10 and SLEEK allows 5 because each SLEEK
    answer performs two model calls.
    """

    strategy: str
    repetitions: int
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    book_key: str = DEFAULT_BOOK_KEY

    @property
    def shot_mode(self) -> str:
        return "Zero-Shot"

    @property
    def database_shot_mode(self) -> str:
        return "zero_shot"

    @property
    def selected_book_keys(self) -> tuple[str, ...]:
        return (self.book_key,)

    @property
    def attempts_per_strategy(self) -> int:
        return QUESTIONS_PER_BOOK

    @property
    def attempts_per_prompt(self) -> int:
        return self.repetitions

    @property
    def total_mutations(self) -> int:
        return QUESTIONS_PER_BOOK

    @property
    def scored_generations(self) -> int:
        return self.total_mutations * self.repetitions

    @property
    def model_calls(self) -> int:
        calls_per_answer = 2 if self.strategy == SLEEK_MODE else 1
        return self.scored_generations * calls_per_answer

    def validate(self, available_modes: Optional[Sequence[str]] = None) -> None:
        modes = tuple(available_modes) if available_modes is not None else PROBE_MODES
        if self.strategy not in modes or self.strategy not in PROBE_MODES:
            raise GameValidationError("Select a supported knowledge-memorization mode.")
        if self.book_key not in BOOK_KEYS:
            raise GameValidationError(
                "Select one of the five Knowledge Memorization books."
            )
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int):
            raise GameValidationError("Runs per question must be an integer.")
        maximum = max_runs_for_mode(self.strategy)
        if not MIN_RUNS <= self.repetitions <= maximum:
            raise GameValidationError(
                f"{self.strategy} supports between {MIN_RUNS} and {maximum} runs."
            )
        if isinstance(self.temperature, bool) or isinstance(self.top_p, bool):
            raise GameValidationError("Sampling values must be numeric.")
        try:
            temperature = float(self.temperature)
            top_p = float(self.top_p)
        except (TypeError, ValueError) as exc:
            raise GameValidationError("Sampling values must be numeric.") from exc
        if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
            raise GameValidationError(
                f"Temperature must be between {MIN_TEMPERATURE:g} and "
                f"{MAX_TEMPERATURE:g}."
            )
        if not MIN_TOP_P <= top_p <= MAX_TOP_P:
            raise GameValidationError(
                f"Top-p must be between {MIN_TOP_P:g} and {MAX_TOP_P:g}."
            )


# Keep the public name used by the page and persistence layer while ensuring
# Challenge 3 no longer inherits Game 2's persuasion-specific validation.
GameConfig = KnowledgeGameConfig


@dataclass(frozen=True)
class KnowledgeAnswer:
    question: str
    ground_truth: str
    response: str
    token_f1: float
    precision: float
    recall: float

    @property
    def rouge_l(self) -> float:
        """Compatibility alias for the shared competition database score column."""

        return self.token_f1


@dataclass(frozen=True)
class KnowledgeStageOneResult:
    answers: Sequence[KnowledgeAnswer]
    temperature: float
    top_p: float
    book_key: str

    @property
    def token_f1(self) -> float:
        return fmean(answer.token_f1 for answer in self.answers) if self.answers else 0.0

    @property
    def precision(self) -> float:
        return fmean(answer.precision for answer in self.answers) if self.answers else 0.0

    @property
    def recall(self) -> float:
        return fmean(answer.recall for answer in self.answers) if self.answers else 0.0

    @property
    def rouge_l(self) -> float:
        """Compatibility alias for the existing Stage 1 database column."""

        return self.token_f1

    @property
    def response(self) -> str:
        return json.dumps(
            [
                {
                    "question": answer.question,
                    "response": answer.response,
                    "token_f1": answer.token_f1,
                    "precision": answer.precision,
                    "recall": answer.recall,
                }
                for answer in self.answers
            ],
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class KnowledgeGeneration:
    mutation_attempt: int
    prompt_attempt: int
    mutated_prompt: str
    response: str
    metrics: Dict[str, float]
    book_key: str = DEFAULT_BOOK_KEY
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_f1(self) -> float:
        return float(
            self.metrics.get("token_f1", self.metrics.get("rouge_l", 0.0)) or 0.0
        )

    @property
    def precision(self) -> float:
        return float(self.metrics.get("precision", 0.0) or 0.0)

    @property
    def recall(self) -> float:
        return float(self.metrics.get("recall", 0.0) or 0.0)

    @property
    def rouge_l(self) -> float:
        """Compatibility alias: Game 3 stores official Token F1 in rouge_l."""

        return self.token_f1


ScoredGeneration = KnowledgeGeneration


@dataclass(frozen=True)
class StageTwoResult:
    config: KnowledgeGameConfig
    generations: Sequence[KnowledgeGeneration]

    @property
    def scores(self) -> List[float]:
        return [generation.token_f1 for generation in self.generations]

    @property
    def max_token_f1(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def avg_token_f1(self) -> float:
        return fmean(self.scores) if self.scores else 0.0

    @property
    def avg_precision(self) -> float:
        return (
            fmean(generation.precision for generation in self.generations)
            if self.generations
            else 0.0
        )

    @property
    def avg_recall(self) -> float:
        return (
            fmean(generation.recall for generation in self.generations)
            if self.generations
            else 0.0
        )

    @property
    def max_score(self) -> float:
        """UI/storage-neutral alias for the peak Token F1."""

        return self.max_token_f1

    @property
    def avg_score(self) -> float:
        """UI/storage-neutral alias for the mean Token F1."""

        return self.avg_token_f1

    @property
    def max_rouge_l(self) -> float:
        return self.max_token_f1

    @property
    def avg_rouge_l(self) -> float:
        return self.avg_token_f1

    @property
    def best_generation(self) -> Optional[KnowledgeGeneration]:
        if not self.generations:
            return None
        return max(self.generations, key=lambda generation: generation.token_f1)


def list_probe_modes() -> List[str]:
    return list(PROBE_MODES)


def _completion() -> Callable[..., Any]:
    from src.direct_recall.comparison import get_llm_completion

    return get_llm_completion


def _completion_text(result: Any) -> str:
    value = result[0] if isinstance(result, tuple) else result
    text = str(value or "").strip()
    if not text:
        raise GameRunError("The model returned an empty response.")
    if text.lower().startswith("error"):
        raise GameRunError(text)
    return text


def _standard_prompt(question: str) -> str:
    """Return the exact prompt used by Knowledge Memorization Standard Q/A."""

    return (
        "Answer the following question concisely and accurately:\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _metrics(question: str, ground_truth: str, response: str) -> Dict[str, float]:
    evaluation = evaluate_qa_comparison(question, ground_truth, response)
    token_f1 = float(evaluation["f1"])
    return {
        "token_f1": token_f1,
        "precision": float(evaluation["precision"]),
        "recall": float(evaluation["recall"]),
        # The shared competition tables call this column rouge_l. Game 3's
        # official meaning is Token F1, so retain an explicit compatibility key.
        "rouge_l": token_f1,
    }


def _answer_standard(
    question: str,
    api_key: str,
    temperature: float,
    top_p: float,
    completion_fn: Callable[..., Any],
) -> str:
    response = answer_question_with_llm(
        question,
        api_key,
        GAME_MODEL,
        GAME_PROVIDER,
        temperature=temperature,
        top_p=top_p,
        max_tokens=150,
        completion_fn=completion_fn,
    )
    return _completion_text(response)


def _answer_sleek(
    question: str,
    api_key: str,
    temperature: float,
    top_p: float,
    completion_fn: Callable[..., Any],
) -> tuple[str, Dict[str, Any]]:
    """Run the real two-call SLEEK workflow from Knowledge Memorization."""

    sub_questions = decompose_question(
        question=question,
        api_key=api_key,
        model_name=GAME_MODEL,
        provider=GAME_PROVIDER,
        temperature=temperature,
        top_p=top_p,
        completion_fn=completion_fn,
    )
    cot_result = run_cot_reasoning(
        original_question=question,
        sub_questions=sub_questions,
        api_key=api_key,
        model_name=GAME_MODEL,
        provider=GAME_PROVIDER,
        temperature=temperature,
        top_p=top_p,
        completion_fn=completion_fn,
    )
    response = _completion_text(cot_result.get("final_answer"))
    return response, {
        "sub_questions": sub_questions,
        "sub_question_answers": cot_result.get("sub_question_answers", []),
        "cot_reasoning": cot_result.get("cot_reasoning", ""),
    }


def _validate_stage_one(book_key: str, temperature: float, top_p: float) -> None:
    config = KnowledgeGameConfig(
        strategy=STANDARD_MODE,
        repetitions=1,
        temperature=temperature,
        top_p=top_p,
        book_key=book_key,
    )
    config.validate()


def run_stage_one(
    api_key: str,
    *,
    book_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    completion_fn: Optional[Callable[..., Any]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> KnowledgeStageOneResult:
    """Run one Standard answer for each of the selected book's five questions."""

    if not str(api_key or "").strip():
        raise GameRunError("The shared competition API key is not configured.")
    _validate_stage_one(book_key, temperature, top_p)
    pairs = get_question_bank(book_key)
    if len(pairs) != QUESTIONS_PER_BOOK:
        raise GameRunError("The frozen five-question baseline is incomplete.")

    completion = completion_fn or _completion()
    answers: List[KnowledgeAnswer] = []
    for question_index, pair in enumerate(pairs, start=1):
        response = _answer_standard(
            pair["question"],
            api_key,
            float(temperature),
            float(top_p),
            completion,
        )
        metrics = _metrics(pair["question"], pair["answer"], response)
        answers.append(
            KnowledgeAnswer(
                question=pair["question"],
                ground_truth=pair["answer"],
                response=response,
                token_f1=metrics["token_f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
            )
        )
        if on_progress is not None:
            on_progress(question_index, QUESTIONS_PER_BOOK)

    if len(answers) != QUESTIONS_PER_BOOK:
        raise GameRunError("The baseline was incomplete and was not recorded.")
    return KnowledgeStageOneResult(
        answers=answers,
        temperature=float(temperature),
        top_p=float(top_p),
        book_key=book_key,
    )


def run_stage_two(
    api_key: str,
    config: KnowledgeGameConfig,
    *,
    completion_fn: Optional[Callable[..., Any]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> StageTwoResult:
    """Run all five questions for the selected number of Standard/SLEEK runs."""

    if not str(api_key or "").strip():
        raise GameRunError("The shared competition API key is not configured.")
    if not isinstance(config, KnowledgeGameConfig):
        raise GameValidationError("Use the dedicated Knowledge Game configuration.")
    config.validate()
    pairs = get_question_bank(config.book_key)
    if len(pairs) != QUESTIONS_PER_BOOK:
        raise GameRunError("The frozen five-question benchmark is incomplete.")

    completion = completion_fn or _completion()
    generations: List[KnowledgeGeneration] = []
    completed = 0
    total = config.scored_generations
    for question_index, pair in enumerate(pairs, start=1):
        for run_index in range(1, config.repetitions + 1):
            trace: Dict[str, Any] = {}
            if config.strategy == SLEEK_MODE:
                response, trace = _answer_sleek(
                    pair["question"],
                    api_key,
                    float(config.temperature),
                    float(config.top_p),
                    completion,
                )
                prompt = f"SLEEK two-call probe: {pair['question']}"
            else:
                response = _answer_standard(
                    pair["question"],
                    api_key,
                    float(config.temperature),
                    float(config.top_p),
                    completion,
                )
                prompt = _standard_prompt(pair["question"])

            metrics = _metrics(pair["question"], pair["answer"], response)
            trace = {"ground_truth": pair["answer"], **trace}
            generations.append(
                KnowledgeGeneration(
                    mutation_attempt=question_index,
                    prompt_attempt=run_index,
                    mutated_prompt=prompt,
                    response=response,
                    metrics=metrics,
                    book_key=config.book_key,
                    trace=trace,
                )
            )
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

    if len(generations) != total:
        raise GameRunError("The enhanced probe was incomplete and was not submitted.")
    return StageTwoResult(config=config, generations=generations)


__all__ = [
    "GameConfig",
    "GameRunError",
    "GameValidationError",
    "KnowledgeAnswer",
    "KnowledgeGameConfig",
    "KnowledgeGeneration",
    "KnowledgeStageOneResult",
    "ScoredGeneration",
    "StageTwoResult",
    "get_book_title",
    "get_question_bank",
    "list_game_books",
    "list_probe_modes",
    "run_stage_one",
    "run_stage_two",
]