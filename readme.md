<div align="center">

# 🕵️ Copyright Detective

**A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks**

🎉 **Accepted to ACL 2026 System Demonstrations!** 🎉

[![Paper](https://img.shields.io/badge/ACL_2026_Demo-Paper-b31b1b.svg?logo=googlescholar&logoColor=white)](https://aclanthology.org/2026.acl-demo.2.pdf)
[![Project Page](https://img.shields.io/badge/Project-Homepage-blue)](https://changhu73.github.io/projects/copyright-detective)
[![Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B)](https://copyright-detective.streamlit.app)
[![Video](https://img.shields.io/badge/YouTube-Video-FF0000)](https://youtu.be/8jYL4CJ-jks)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://github.com/zdh2292390/Copyright-Detective/blob/main/LICENSE)

*For more details, please refer to [Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks](https://aclanthology.org/2026.acl-demo.2.pdf)*

</div>

<img width="2542" height="1348" alt="overview" src="https://github.com/user-attachments/assets/26485125-843f-461a-9f4b-3f92d622e23e" />

---

## How to Run

Analyze potential text copyright infringement in LLM applications. Follow the steps below to set up your environment and run the application of Copyright-Detective.

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
  --tensor-parallel-size 4 \
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

---

## 🌐 vLLM Multi-Model Serving (2 Models) — Setup & Usage (Optional)

If you want to use **multiple vLLM models at the same time**, you can route them through a single public endpoint using **Caddy + ngrok**.

### 1) Server: Start Two vLLM Instances (Localhost)

```bash
# Model 1
vllm serve /path/to/model1 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key YOUR_API_KEY \
  --served-model-name 1

# Model 2
vllm serve /path/to/model2 \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key YOUR_API_KEY \
  --served-model-name 2
```

### 2) Server: Install Caddy

```bash
wget https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz
tar -zxvf caddy_2.8.4_linux_amd64.tar.gz
chmod +x caddy
./caddy version
```

### 3) Server: Configure `Caddyfile` (Reverse Proxy by Path)

Create / edit `Caddyfile`:

```caddyfile
:8081 {
    handle_path /m1/* {
        reverse_proxy 127.0.0.1:8000
    }

    handle_path /m2/* {
        reverse_proxy 127.0.0.1:8001
    }
}
```

Run Caddy:

```bash
./caddy run --config Caddyfile > caddy.log 2>&1 &
```

### 4) Server: Expose via ngrok

```bash
./ngrok http 8081
```

Now you can access:

- **Model 1** via: `https://xxxx-xxx-xxx.ngrok-free.dev/m1/v1`
- **Model 2** via: `https://xxxx-xxx-xxx.ngrok-free.dev/m2/v1`

> **Note:** Keep the `/v1` suffix for OpenAI-compatible APIs served by vLLM.

---

## 🧠 Unlearning Detection — (Deployment)

This module runs a lightweight deployment service on your **remote server**, then exposes it to your local Streamlit app via a temporary Cloudflare tunnel.


### 1) Server (Remote): Start the Deployment Service

export your key:
```bash
export YOUR_API_KEY="your_api_key"
```

Before running [unlearning_deploy.py](backend/unlearning_deploy.py), review the [remote deployment safety notes](backend/notes.md).

Once the service is running and the tunnel is up, you can use **Representational Analysis (Unlearning Detection)** directly from the app UI.

### 2) Server (Remote): Install `cloudflared` and Start a Quick Tunnel

On your server, download and start Cloudflare Tunnel:

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
./cloudflared-linux-amd64 tunnel --url http://localhost:1234
```

You should see output similar to:

```bash
Requesting new quick Tunnel on trycloudflare.com...
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://cool-server-link.trycloudflare.com                                                |
+--------------------------------------------------------------------------------------------+
```

Copy the generated `https://*.trycloudflare.com` link and paste it into the app:

- **Deployment Agent URL** → `https://cool-server-link.trycloudflare.com`

- **KEY** → `your_api_key`

> **Note:** It may take a short time before the tunnel becomes reachable.

### 3) App: Set Model Paths (Absolute Paths on the Server)

In **Representational Analysis (Unlearning Detection)**, provide the **absolute paths on the deployment server**:

- **Reference model path** → `/absolute/path/to/reference_model`
- **Unlearned model path** → `/absolute/path/to/unlearned_model`

> The paths must be server-local absolute paths (not your local machine paths).


---

> **⚠️ Disclaimer: Non-Commercial Research Project**
>
> This project, **Copyright Detective**, is an independent academic research initiative.
>
> It is **NOT** affiliated with, endorsed by, or connected to **"The Copyright Detective"** or any of its associated commercial entities.
>
> This software is strictly **non-commercial**, created solely for **research purposes**. No commercial gain is intended or realized.
>
> If you are a trademark owner and have any concerns, please contact the author directly at **dhzhangai@gmail.com** before taking further action. I am open to cooperation to resolve any potential confusion.

---

## 📝 Citation
If you find this project useful for your research, please consider citing our paper:

```bibtex
@inproceedings{zhang-etal-2026-copyright,
    title = "Copyright Detective: A Forensic System to Evidence {LLM}s Flickering Copyright Leakage Risks",
    author = "Zhang, Guangwei  and
      Zhu, Jianing  and
      Qian, Cheng  and
      Gong, Neil Zhenqiang  and
      Mihalcea, Rada  and
      Xu, Zhaozhuo  and
      He, Jingrui  and
      Ma, Jiaqi W.  and
      Xiao, Chaowei  and
      Li, Bo  and
      Abbasi, Ahmed  and
      Lee, Dongwon  and
      Ji, Heng  and
      Zhang, Denghui",
    editor = "Durrett, Greg  and
      Jian, Ping",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-demo.2/",
    doi = "10.18653/v1/2026.acl-demo.2",
    pages = "14--26",
    ISBN = "979-8-89176-392-0",
    abstract = "We present \textbf{Copyright Detective}, the first interactive forensic system for detecting, analyzing, and visualizing potential copyright risks in LLM outputs. The system treats copyright infringement versus compliance as an \textbf{evidence discovery} process rather than a static classification task due to the complex nature of copyright law. It integrates multiple detection paradigms, including content recall testing, paraphrase-level similarity analysis, persuasive jailbreak probing, and unlearning verification, within a unified and extensible framework. Through interactive prompting, response collection, and iterative workflows, our system enables systematic auditing of verbatim memorization and paraphrase-level leakage, supporting responsible deployment and transparent evaluation of LLM copyright risks even with black-box access. In our experiments with GPT-4o-mini, we demonstrate that the specific persuasive strategy ``Pathos'' shifts the leakage distribution from about 0.1 (ROUGE-L) to 0.7. Our live system is hosted on \href{https://copyright-detective.streamlit.app}{Streamlit server}, with a \href{https://youtu.be/z9Lh4kNDHiM}{demonstration video} included as supplementary material."
}
````
