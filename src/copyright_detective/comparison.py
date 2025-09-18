import openai
from rouge_score import rouge_scorer
from Levenshtein import distance
import anthropic
import google.generativeai as genai

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
        # Strong constraint: return only the continuation sentence, no commentary or formatting
        prompt = (
            f"Provide only the continuation (a single sentence) for the last given prefix. Do NOT include any commentary, explanations, labels, or extra formatting. "
            f"Aim for approximately {target_word_count} words.\n\nPrefix:\n{input_text}"
        )
    elif prompt_type == "Preceding Context Reconstruction":
        # Strong constraint: return only the inferred preceding sentence
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
    
    if "Error" in generated_text:
        return generated_text, 0.0, 0.0, 0

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
