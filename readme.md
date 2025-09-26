# How to Run Copyright Detective

Analyze potential text copyright infringement in LLM application.

Follow these steps to set up your environment and run the application.

/home/changhu/miniconda3/envs/copyright-detective/bin/python /home/changhu/Copyright-Detective/


## 1. Create a Conda Environment

It is highly recommended to use a Conda environment to manage project dependencies.

Open your terminal and run the following command in the project's root directory to create a new environment named `copyright-detective`:

```bash
conda create --name copyright-detective python=3.9 -y
```

This will create a new Conda environment with Python 3.9.

## 2. Activate the Conda Environment

Before you can install packages or run the app, you need to activate the environment.

```bash
conda activate copyright-detective
```

Your terminal prompt should now show `(copyright-detective)` at the beginning, indicating that the environment is active.

## 3. Install Required Packages

With the virtual environment active, install all the necessary libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install `streamlit`, `openai`, `PyPDF2`, and `rouge-score`.

## 4. Run the Streamlit Application

Now you are ready to start the application. Run the following command:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in a new tab in your default web browser.

You can now interact with the "Copyright Detective" tool. Make sure you have a valid OpenAI API key to use the model-based features.

### Custom continuation prompts

When analysing text snippets you can now choose **Custom Prompt** from the continuation method selector. This lets you paste a full instruction template tailored to your experiment. Use the `{input_text}` placeholder wherever the user-supplied snippet should appear. You can also optionally reference `{word_count}` or `{char_count}` to mirror the target length derived from your ground truth. The preview panel will render the final prompt with any available values filled in so you can double-check the framing before running inference.

The same **Custom Prompt** option is available during whole-PDF analysis. Whatever template you supply is injected into every chunk before the model is asked to continue it, so you can drive consistent jailbreak persuasion experiments across long-form documents without losing control over the framing.

## Jailbreak Persuasion Probe

This new page helps you design and evaluate jailbreak/persuasion prompts in a safety-first manner. It analyzes prompts for risky indicators (e.g., location-based extraction or exact-length replication) and provides compliant alternatives and refusal templates. The probe only performs meta-analysis and does not request or display copyrighted text.

The page also includes a small, curated library of well-known jailbreak prompts (for analysis only). You can filter, preview, and load any into the evaluator textbox to see risk detections — again, it never asks the model to output copyrighted content.

## Unlearning Detection

The **Unlearning Detection** workspace combines two complementary probes. Use the tabs at the top of the workspace to switch between them so that their controls and outputs stay neatly separated:

- **Prompt-Based Probes** reuse the persuasion strategies from snippet/PDF analysis to see whether a model can still recreate or summarise a target passage. The tool now focuses on surfacing each model response directly—no similarity thresholds or ground-truth references are required.
- **Membership Inference (Perplexity Probe)** estimates whether the reference text still lives in the model's training data. The app slices both passages into 50–200 token windows (falling back to word windows if the tokenizer is unavailable), samples per-token log-probabilities with `echo=True` completions (currently supported for OpenAI models), computes average perplexity via $\text{PPL} = e^{-\overline{\log p}}$, and reports a "training trace" score for each chunk. A configurable ΔPPL threshold, together with Welch's t-test and Kolmogorov–Smirnov comparisons, highlights statistically significant gaps between reference and control distributions.

To run the perplexity probe, supply both the reference text and an equivalent-length control passage. You can adjust token window size, number of sampled windows, and the perplexity gap required to flag memorisation. When SciPy is unavailable the statistical tests degrade gracefully to analytical approximations, and the UI will annotate any fallbacks or tokenizer limitations.

## PDF Analysis Notes

- Chunk size is specified in words. During Whole PDF Analysis, the Generated Text produced for each chunk is enforced to be exactly the same number of words as the selected chunk size. This ensures fair, length-controlled comparisons across chunks.
- You can optionally select a persuasion framing (e.g., Role-Playing or Lost Manuscript) when running Whole PDF Analysis. The app will use that strategy for every chunk, mirroring the Text Snippet workflows and making it easier to probe jailbreak-style continuations.
- If you pick **Custom Prompt**, the PDF analyzer will apply your template to every chunk. Ensure you include `{input_text}` (and optionally `{word_count}` or `{char_count}`) so the chunk content and length guidance are inserted automatically.
- Temperature and Top-P controls are now available for PDF runs so you can match the sampling behaviour to your text experiments.
- Use the **Ranks to Display** control to decide how many of the highest-scoring chunks are surfaced in the final summary.
