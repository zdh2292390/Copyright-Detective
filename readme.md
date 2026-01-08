# 🕵️ Copyright Detective

**Copyright Detective** is a forensic tool designed to analyze potential copyright infringement in Large Language Models (LLMs). It provides a suite of probes to detect memorization, test robustness against adversarial attacks, and verify unlearning efficacy.

<img width="3220" height="1840" alt="overview" src="https://github.com/user-attachments/assets/268b5366-d560-4bd8-8047-5613f70a2c2a" />

---

## How to Run

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

---

## 🌐 Sidebar — vLLM Model Setup & Usage

### Scenario A — Run the App Locally, Serve vLLM Remotely (Private Network / No Public Exposure, you can also use the method in scenario B)

Use this setup when your **vLLM model is deployed on a remote server**, but you run **this project locally** (e.g., on your laptop). The vLLM endpoint is reachable via your internal network/VPN/SSH tunnel, and you **do not** need to expose it to the public internet.

#### 1) Server (Remote): Start vLLM

Run on the remote server:

```bash
vllm serve YOUR_MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --api-key YOUR_API_KEY \
  --served-model-name YOUR_MODEL_NAME
```

#### 2) Client (Local): Connect to the Server (Optional)

If you need to log into the remote server:

```bash
ssh YOUR_USER@YOUR_SERVER_IP
```

#### 3) App (Local): Configure Model Selection

In the Streamlit sidebar, go to **✨ Model Selection** and set:

- **Provider:** `Local vLLM`
- **Input Model Choice:** `YOUR_MODEL_ID` *(optional; used only as an identifier/label)*

> **Note:** Ensure your API key is configured correctly. For GPU-based features, install `torch` with CUDA support on the machine that runs GPU workloads.

**Responsible use:** This workspace is intended for auditing and defense research. Handle all generated content responsibly, follow institutional review policies, and do not redeploy harmful variants outside controlled evaluation environments.

---

### Scenario B — Run vLLM Remotely and Access It from the Web (Expose via ngrok)

Use this setup when you want to **expose your remote vLLM endpoint** to the public internet (for temporary demos or remote access). This uses **ngrok** as a secure tunnel.

#### 1) Server (Remote): Start vLLM (Bind to localhost)

On the deployment server, run:

```bash
vllm serve YOUR_MODEL_PATH \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key YOUR_API_KEY \
  --served-model-name YOUR_MODEL_NAME
```

Wait until you see:

```bash
Uvicorn running on http://127.0.0.1:8000
```

#### 2) Server (Remote): Install and Configure ngrok

```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
./ngrok config add-authtoken <YOUR_TOKEN>
```

#### 3) Server (Remote): Start ngrok

```bash
./ngrok http 8000
```

You should see output like:

```bash
Forwarding https://xxxx-xxx-xxx.ngrok-free.dev -> http://localhost:8000
```

#### 4) App: Set the Base URL

In the app settings, set **Base URL** to:

```text
https://xxxx-xxx-xxx.ngrok-free.dev/v1
```

> **Tip:** The `/v1` suffix is required for OpenAI-compatible APIs served by vLLM.
