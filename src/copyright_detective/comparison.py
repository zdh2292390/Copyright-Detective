import openai
from rouge_score import rouge_scorer

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

def compare_texts(upper_text, lower_text, api_key, model_name, provider="OpenAI"):
    """
    Generates text based on the upper context and compares it to the lower context.
    """
    target_word_count = len(lower_text.split())
    prompt = f"Based on the following text, continue writing the next paragraph. The continuation should be approximately {target_word_count} words long.\n\n{upper_text}"
    generated_text = get_llm_completion(prompt, api_key, model_name, provider)
    
    if "Error" in generated_text:
        return generated_text, 0.0, 0.0

    rouge_score = calculate_rouge_score(lower_text, generated_text)
    jaccard_index = calculate_jaccard_index(lower_text, generated_text)
    return generated_text, rouge_score, jaccard_index
