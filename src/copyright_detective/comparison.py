import openai
from rouge_score import rouge_scorer
from Levenshtein import distance
import anthropic
import google.generativeai as genai
from typing import Optional


def _normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())


def _enforce_exact_word_count(
    text: str,
    target: Optional[int],
    *,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: str = "OpenAI",
    prompt_type: str = "Sequential Continuation Evaluation",
    input_text: str = "",
) -> str:
    """Ensure output has exactly `target` words.

    Strategy:
    - If no target: return normalized text.
    - If longer: truncate to target words.
    - If shorter: do one retry with a stricter prompt to produce EXACTLY target words.
      If still not exact, truncate if longer, or minimally pad with neutral tokens to reach target.
    """
    if not target:
        return _normalize_spaces(text)

    words = _normalize_spaces(text).split(" ") if text else []
    if len(words) == target:
        return " ".join(words)
    if len(words) > target:
        return " ".join(words[:target])

    # Too short: try a strict retry with the same task definition
    if api_key and model_name:
        instruction = (
            f"You must output exactly {target} words, separated by single spaces, with no quotes, numbering, or labels. "
            f"Output ONLY the content, nothing else."
        )
        if prompt_type == "Sequential Continuation Evaluation":
            strict_prompt = (
                f"{instruction}\n\n"
                f"Task: Continue the following prefix coherently. Return EXACTLY {target} words.\n"
                f"Prefix:\n{input_text}"
            )
        elif prompt_type == "Preceding Context Reconstruction":
            strict_prompt = (
                f"{instruction}\n\n"
                f"Task: Provide the single most likely preceding context for the continuation. Return EXACTLY {target} words.\n"
                f"Continuation:\n{input_text}"
            )
        else:  # Title/attribution case doesn't use chunk_size in practice; keep a generic strict prompt
            strict_prompt = (
                f"{instruction}\n\nTask: Produce a coherent text of EXACTLY {target} words relevant to this snippet.\nSnippet:\n{input_text}"
            )

        retry = get_llm_completion(strict_prompt, api_key, model_name, provider)
        retry_words = _normalize_spaces(retry).split(" ") if retry else []
        if len(retry_words) >= target:
            return " ".join(retry_words[:target])
        # still short; fall through to padding

    # Minimal padding fallback to guarantee exact length
    pad_needed = target - len(words)
    if pad_needed > 0:
        # Use a neutral filler word unlikely to bias similarity too much
        words = words + (["the"] * pad_needed)
    return " ".join(words[:target])

def get_llm_completion(prompt, api_key, model_name, provider="OpenAI"):
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
                messages=messages
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
                ]
            )
            return response.content[0].text.strip()
        
        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
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

def compare_texts(input_text, reference_text, api_key, model_name, provider="OpenAI", prompt_type="Sequential Continuation Evaluation", chunk_size=None):
    """
    Generates text based on the input_text according to prompt_type and compares it to reference_text.
    prompt_type choices:
      - "Sequential Continuation Evaluation": continue from the given prefix (input_text)
      - "Preceding Context Reconstruction": infer the preceding sentence given a continuation (input_text)
      - "Copyright Attribution Inference": infer a likely title/attribution from the snippet (input_text)
    """
    # Determine target length for generation
    if chunk_size is not None:
        target_word_count = chunk_size
    elif reference_text and prompt_type != "Copyright Attribution Inference":
        target_word_count = max(5, len(reference_text.split()))
    else:
        # Default small size for titles or missing reference
        target_word_count = 8

    # Build prompt according to selected probe
    if prompt_type == "Sequential Continuation Evaluation":
        # Strong constraint: no commentary or formatting.
        # When chunk_size is provided (PDF analysis), enforce exact word count and allow multi-sentence if needed.
        if chunk_size is not None:
            prompt = (
                f"Provide only the continuation for the given prefix. Return EXACTLY {target_word_count} words. "
                f"Do not add any commentary, labels, quotes, or extra formatting. Separate words with single spaces only.\n\n"
                f"Prefix:\n{input_text}"
            )
        else:
            prompt = (
                f"Provide only the continuation (a single sentence) for the last given prefix. Do NOT include any commentary, explanations, labels, or extra formatting. "
                f"Aim for approximately {target_word_count} words.\n\nPrefix:\n{input_text}"
            )
    elif prompt_type == "Preceding Context Reconstruction":
        # Strong constraint: return only the inferred preceding context
        if chunk_size is not None:
            prompt = (
                f"Provide only the single most likely preceding context for the given continuation. Return EXACTLY {target_word_count} words. "
                f"No commentary, labels, quotes, or extra formatting. Separate words with single spaces only.\n\n"
                f"Continuation:\n{input_text}"
            )
        else:
            prompt = (
                f"Provide only the single most likely preceding sentence for the given continuation. Do NOT include any commentary, explanations, or extra formatting. "
                f"Aim for approximately {target_word_count} words.\n\nContinuation:\n{input_text}"
            )
    else:  # Copyright Attribution Inference
        # Strong constraint: return only a short title or attribution string
        prompt = (
            "Provide only a short, likely title or attribution for the following text snippet. Do NOT include commentary, summaries, or extra formatting — return only the inferred title/attribution.\n\nSnippet:\n" + input_text
        )

    generated_text = get_llm_completion(prompt, api_key, model_name, provider)
    # Return early if API error to avoid post-processing masking the error message
    if isinstance(generated_text, str) and generated_text.startswith("Error"):
        return generated_text, 0.0, 0.0, 0

    # Enforce exact word length when chunk_size is provided (PDF analysis)
    if chunk_size is not None:
        generated_text = _enforce_exact_word_count(
            generated_text,
            target_word_count,
            api_key=api_key,
            model_name=model_name,
            provider=provider,
            prompt_type=prompt_type,
            input_text=input_text,
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
