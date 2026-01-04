# 🕵️ Copyright Detective

**Copyright Detective** is a forensic tool designed to analyze potential copyright infringement in Large Language Models (LLMs). It provides a suite of probes to detect memorization, test robustness against adversarial attacks, and verify unlearning efficacy.

<img width="3220" height="1840" alt="overview" src="https://github.com/user-attachments/assets/268b5366-d560-4bd8-8047-5613f70a2c2a" />

# How to Run

Analyze potential text copyright infringement in LLM applications. Follow the steps below to set up your environment and run the application.

## 🚀 Quick Start (Local)

```bash
# Create and activate environment
conda create -n copyright-detective python=3.11 -y
conda activate copyright-detective

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## 🌐 Remote vLLM Deployment (Optional)

Use this option if your vLLM model is deployed on a separate server.

### 1) Server-side: Start vLLM

Run the following command on the remote server to start the vLLM service:

```bash
vllm serve YOUR_MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --api-key YOUR_API_KEY
```

### 2) Client-side: Connect (Optional)

If you need to log into the remote server:

```bash
ssh YOUR_USER@YOUR_SERVER_IP
```

### 3) App: Configure Model Selection

In the Streamlit sidebar, go to **✨ Model Selection** and set:

- **Provider:** `Local vLLM`
- **Input Model Path:** `YOUR_MODEL_PATH` (must match the path used on the server)

> **Note:** Ensure you have valid API keys ready. For GPU-based features, make sure `torch` is installed with CUDA support.



This workspace is intended for auditing and defense research. Handle all generated content responsibly, follow institutional review policies, and avoid redeploying harmful mutations outside controlled evaluations.
