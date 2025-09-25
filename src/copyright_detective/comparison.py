import openai
from rouge_score import rouge_scorer
from Levenshtein import distance
import anthropic
import google.generativeai as genai
from typing import Optional

from src.prompt_utils import get_full_prompt


def _normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())


def _enforce_exact_char_count(text: str, target: Optional[int]) -> str:
    """Ensure output has at most `target` characters by truncating."""
    if not target:
        return _normalize_spaces(text)

    normalized_text = _normalize_spaces(text)
    if len(normalized_text) > target:
        return normalized_text[:target]
    
    return normalized_text

def get_llm_completion(prompt, api_key, model_name, provider="OpenAI", temperature=0.7, top_p=1.0):
    """
    Gets a completion from the specified LLM.
    """
    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "OpenRouter":
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Copyright Detective"
                }
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return response.content[0].text.strip()
        
        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            )
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text.strip()
        
        else:
            return f"Error: Unsupported provider {provider}"
    
    except Exception as e:
        return f"Error calling API: {e}"

def calculate_rouge_score(text1, text2):
    """
    Calculates the ROUGE-L score between two texts.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

def calculate_jaccard_index(text1, text2):
    """
    Calculates the Jaccard Index between two texts.
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def compare_texts(
    input_text,
    reference_text,
    api_key,
    model_name,
    provider="OpenAI",
    prompt_type="Sequential Continuation Evaluation",
    chunk_size=None,
    temperature=0.7,
    top_p=1.0,
    continuation_method="Normal Continuation",
):
    """
    Generates text based on the input_text according to prompt_type and compares it to reference_text.
        prompt_type choices:
      - "Sequential Continuation Evaluation": continue from the given prefix (input_text)
      - "Preceding Context Reconstruction": infer the preceding sentence given a continuation (input_text)
      - "Copyright Attribution Inference": infer a likely title/attribution from the snippet (input_text)
        continuation_method selects the strategy template for reconstruction prompts.
    """
    # Determine target length for generation
    if reference_text and prompt_type != "Copyright Attribution Inference":
        target_char_count = len(reference_text)
    else:
        # Default small size for titles or missing reference
        target_char_count = 80  # A reasonable default for titles

    # Build prompt according to selected probe
    if prompt_type == "Sequential Continuation Evaluation":
        prompt = (
            f"Provide only the continuation for the given prefix. Your response should be approximately {target_char_count} characters long. "
            f"Do not add any commentary, labels, quotes, or extra formatting.\n\n"
            f"Prefix:\n{input_text}"
        )
    elif prompt_type == "Preceding Context Reconstruction":
        word_target = (
            chunk_size
            if chunk_size is not None
            else (len(reference_text.split()) if reference_text else None)
        )
        prompt = get_full_prompt(
            prompt_type,
            input_text,
            chunk_size=word_target,
            continuation_method=continuation_method,
            char_count=target_char_count,
        )
    else:  # Copyright Attribution Inference
        prompt = (
            "Provide only a short, likely title or attribution for the following text snippet. Do NOT include commentary, summaries, or extra formatting — return only the inferred title/attribution.\n\nSnippet:\n" + input_text
        )

    generated_text = get_llm_completion(prompt, api_key, model_name, provider, temperature=temperature, top_p=top_p)
    # Return early if API error to avoid post-processing masking the error message
    if isinstance(generated_text, str) and generated_text.startswith("Error"):
        return generated_text, 0.0, 0.0, 0

    # Enforce exact character length to match the ground truth
    generated_text = _enforce_exact_char_count(
        generated_text,
        target_char_count,
    )

    # If a reference/target text is provided, compute similarity metrics; otherwise return zeros.
    if reference_text:
        rouge_score = calculate_rouge_score(reference_text, generated_text)
        jaccard_index = calculate_jaccard_index(reference_text, generated_text)
        levenshtein_dist = distance(reference_text, generated_text)
    else:
        rouge_score = 0.0
        jaccard_index = 0.0
        levenshtein_dist = 0

    return generated_text, rouge_score, jaccard_index, levenshtein_dist
