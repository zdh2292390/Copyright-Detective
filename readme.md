# How to Run Copyright Detective

Analyze potential text copyright infringement in LLM application.

Follow these steps to set up your environment and run the application.

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

This will install the baseline libraries (`streamlit`, `openai`, `PyPDF2`, `rouge-score`, etc.).

> **GPU extras:** The new representational analysis probes rely on deep learning stacks. After the base install, add the optional dependencies with `pip install torch transformers scikit-learn` (or your preferred CUDA build of PyTorch). These are already listed in `requirements.txt`, but you may want to install a hardware-specific wheel manually to ensure GPU acceleration.

## 4. Run the Streamlit Application

Now you are ready to start the application. Run the following command:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in a new tab in your default web browser.

You can now interact with the "Copyright Detective" tool. Make sure you have a valid OpenAI API key to use the model-based features.

## Snippet-to-Document Analysis

The primary workspace now bundles both the snippet evaluator and the PDF sweeps behind a single navigation entry. Once you open **Snippet-to-Document Analysis** you'll find two tabs mirroring the Unlearning Detection layout: one for short-form snippets and another for full documents. Each tab preserves the controls you already know, so you can jump between granular spot checks and long-form discovery without leaving the page.

### Custom continuation prompts

Inside the **Text Snippet Analysis** tab you can choose **Custom Prompt** from the continuation method selector. This lets you paste a full instruction template tailored to your experiment. Use the `{input_text}` placeholder wherever the user-supplied snippet should appear. You can also optionally reference `{word_count}` or `{char_count}` to mirror the target length derived from your ground truth. The preview panel will render the final prompt with any available values filled in so you can double-check the framing before running inference.

The same **Custom Prompt** option is available on the **Whole PDF Analysis** tab. Whatever template you supply is injected into every chunk before the model is asked to continue it, so you can drive consistent jailbreak persuasion experiments across long-form documents without losing control over the framing.

### Multi-run diversity diagnostics

When you request multiple inference runs, the results page now augments the similarity stats with **output diversity diagnostics**. The tool aggregates all generated samples, computes their Shannon entropy (both in raw bits and as a percentage of the theoretical maximum), and highlights how much probability mass collapses onto the most common continuation. A ranked table plus bar chart surfaces the top-$k$ variants so you can quickly spot mode collapse or suspiciously stable reproductions that may hint at residual memorisation.

If the entropy stays low or a single continuation dominates the distribution, the interface raises an inline warning to encourage deeper investigation (e.g., bumping temperature or trying alternative prompts).

## Jailbreak Persuasion Probe

This new page helps you design and evaluate jailbreak/persuasion prompts in a safety-first manner. It analyzes prompts for risky indicators (e.g., location-based extraction or exact-length replication) and provides compliant alternatives and refusal templates. The probe only performs meta-analysis and does not request or display copyrighted text.

The page also includes a small, curated library of well-known jailbreak prompts (for analysis only). You can filter, preview, and load any into the evaluator textbox to see risk detections — again, it never asks the model to output copyrighted content.

## Unlearning Detection

The **Unlearning Detection** workspace combines three complementary probes. Use the tabs at the top of the workspace to keep their controls and outputs neatly separated:

- **Prompt-Based Probes** reuse the persuasion strategies from snippet/PDF analysis to measure residual recall through structured summaries and evidence inventories. Rather than nudging the model to continue the text, each probe now asks for audit-style descriptions that expose what, if anything, remains remembered. The tool surfaces every response directly—no similarity thresholds or ground-truth references are required.
- **Membership Inference (Perplexity Probe)** estimates whether the reference text still lives in the model's training data. The app slices both passages into 50–200 token windows (falling back to word windows if the tokenizer is unavailable), samples per-token log-probabilities with `echo=True` completions (currently supported for OpenAI models), computes average perplexity via $\text{PPL} = e^{-\overline{\log p}}$, and reports a "training trace" score for each chunk. A configurable ΔPPL threshold, together with Welch's t-test and Kolmogorov–Smirnov comparisons, highlights statistically significant gaps between reference and control distributions.
- **Representational Analysis (Feature Probes)** compares the hidden-state geometry of the reference and updated models. Pick from Fisher Information, PCA shift/similarity, or linear CKA analyses; supply a list of evaluation prompts along with the reference/deployed checkpoints; and the app will write per-layer PDFs (or a single plot) into your chosen output folder. Optional controls let you point to a CUDA device, adjust batch sizes, and cap tokenizer length.

All representational probes run locally on your hardware. Ensure that the model checkpoints fit in GPU memory (or switch the device to `cpu` if you're validating smaller networks). The UI bundles directory outputs into a downloadable `.zip` when possible, and individual PDFs can be downloaded directly for a quick peek.

### 🛠️ Python API (Representational Toolkit)

You can use the same representational probes directly from Python by importing the unified helper:

```python
from representational_toolkit.analysis import run_feature_analysis

query = [
	"The quick brown fox jumps over the lazy dog.",
	"Unlearning LLMs is an active area of research."
]

# 1) Fisher Information (writes one PDF per transformer layer)
run_feature_analysis(
	feature="fim",
	model_reference_path="Qwen/Qwen2.5-7B",
	model_path="your_own_model_path",
	query=query,
	output_path="./fim_output",   # directory for per-layer PDFs
	device="cuda",
	batch_size=4,
	num_batches=10,
	max_length=128,
)

# 2) PCA Shift (Δ PC1 vs. PC2)
run_feature_analysis(
	feature="pca_shift",
	model_reference_path="Qwen/Qwen2.5-7B",
	model_path="your_own_model_path",
	query=query,
	output_path="./pca_shift.pdf",  # single PDF (or .png)
	device="cuda",
	max_length=128,
)

# 3) PCA Cosine Similarity of PC1
run_feature_analysis(
	feature="pca_sim",
	model_reference_path="Qwen/Qwen2.5-7B",
	model_path="your_own_model_path",
	query=query,
	output_path="./pca_sim.pdf",    # single PDF (or .png)
	device="cuda",
	max_length=128,
)

# 4) Layer-wise Linear CKA
run_feature_analysis(
	feature="cka",
	model_reference_path="Qwen/Qwen2.5-7B",
	model_path="your_own_model_path",
	query=query,
	output_path="./cka.pdf",        # single PDF (or .png)
	device="cuda",
	batch_size=4,
	num_batches=10,
	max_length=128,
)
```

Each probe shares the same signature—only the `feature` flag and `output_path` semantics change (directory for `fim`, single file for PCA/CKA plots). Feel free to switch `device="cpu"` if your GPU memory is limited.

To run the perplexity probe, supply both the reference text and an equivalent-length control passage. You can adjust token window size, number of sampled windows, and the perplexity gap required to flag memorisation. When SciPy is unavailable the statistical tests degrade gracefully to analytical approximations, and the UI will annotate any fallbacks or tokenizer limitations.

### Whole PDF Analysis Notes

- Chunk size is specified in words. During Whole PDF Analysis, the Generated Text produced for each chunk is enforced to be exactly the same number of words as the selected chunk size. This ensures fair, length-controlled comparisons across chunks.
- You can optionally select a persuasion framing (e.g., Role-Playing or Lost Manuscript) when running Whole PDF Analysis. The app will use that strategy for every chunk, mirroring the Text Snippet workflows and making it easier to probe jailbreak-style continuations.
- If you pick **Custom Prompt**, the PDF analyzer will apply your template to every chunk. Ensure you include `{input_text}` (and optionally `{word_count}` or `{char_count}`) so the chunk content and length guidance are inserted automatically.
- Temperature and Top-P controls are now available for PDF runs so you can match the sampling behaviour to your text experiments.
- Use the **Ranks to Display** control to decide how many of the highest-scoring chunks are surfaced in the final summary.
