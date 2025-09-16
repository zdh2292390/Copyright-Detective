import openai
from rouge_score import rouge_scorer
from Levenshtein import distance

def get_llm_completion(prompt, api_key, model_name, provider="OpenAI"):
    """
    Gets a completion from the specified LLM.
    """
    client_params = {
        "api_key": api_key
    }
    if provider == "OpenRouter":
        client_params["base_url"] = "https://openrouter.ai/api/v1"

    client = openai.OpenAI(**client_params)

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if provider == "OpenRouter":
             # Add custom headers for OpenRouter
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "http://localhost", # Replace with your actual site URL
                    "X-Title": "Copyright Detective" # Replace with your actual app name
                }
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            
        return response.choices[0].message.content.strip()
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

def compare_texts(input_text, reference_text, api_key, model_name, provider="OpenAI", prompt_type="Sequential Reproduction Probe"):
    """
    Generates text based on the input_text according to prompt_type and compares it to reference_text.
    prompt_type choices:
      - "Sequential Reproduction Probe": continue from the given prefix (input_text)
      - "Preceding Context Reconstruction": infer the preceding sentence given a continuation (input_text)
      - "Copyright Attribution Elicitation": infer a likely title/attribution from the snippet (input_text)
    """
    # Determine target length for generation
    if reference_text and prompt_type != "Copyright Attribution Elicitation":
        target_word_count = max(5, len(reference_text.split()))
    else:
        # Default small size for titles or missing reference
        target_word_count = 8

    # Build prompt according to selected probe
    if prompt_type == "Sequential Reproduction Probe":
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
    else:  # Copyright Attribution Elicitation
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
