# Notes: Secure Model Deployment & Analysis Server (`deploy.py`)

This document provides **user-facing** guidance for operating the server safely. Please read before use.

---

## Overview

This server exposes endpoints to:

- deploy a model path (`POST /deploy`)
- run **dynamic Python code** against a loaded model (`POST /run_dynamic_code`)
- run representational analyses (e.g., FIM/PCA/CKA) (`POST /run_analysis`)
- poll async task status (`GET /task_status/{task_id}`)
- check service health (`GET /health`)

**All endpoints require the `X-API-Key` header** (as currently implemented in your code).

---

## Critical Security Warnings

### 1) Authenticated Remote Code Execution (RCE) — **High Risk**
This server executes code sent by clients using `exec()` (both in dynamic code execution and analysis-code execution).

- **What this means:** anyone who obtains a valid API key can run arbitrary Python on the server.
- **Impact:** reading files, exfiltrating environment variables/secrets, network calls, installing malware, deleting data, etc.
- **Bottom line:** API key auth reduces *who* can attack, but it does not remove the underlying capability.

**Do not expose this service to the public internet.** Use only in a controlled environment (private network/VPN + IP allowlist).

---

### 2) Concurrency & Cross-Request Data Leakage (Thread Safety)
The server uses:
- global state (`GLOBAL_*`, `CURRENT_*`, `TASKS`)
- background threads (`threading.Thread`)
- global monkey-patching of `transformers.AutoModelForCausalLM.from_pretrained` during analysis

**Risk:** if multiple tasks run concurrently, one task may affect another (wrong model used, incorrect results, crashes, or data mixing between users).

**Recommended usage:** **single-user / low-concurrency**. Prefer **one analysis at a time**.

---

### 3) Sensitive Error Disclosure (Tracebacks)
When execution fails, full Python tracebacks may be returned/stored in task errors.

- **Impact:** reveals filesystem paths, internal module structure, and sometimes sensitive values.

**Production recommendation:** return generic errors to clients; store detailed tracebacks only in server logs.

---

### 4) Denial-of-Service (DoS) via Resource Exhaustion
Request size limits help, but **they do not prevent**:
- infinite loops
- large tensor allocations / GPU OOM
- huge memory allocation
- extremely long computations

**Impact:** service becomes unavailable; host may become unstable.

**Recommendation:** run this service inside a sandbox/container with CPU/memory/GPU limits, and enforce per-task timeouts.

---

### 5) Transport Security (TLS/HTTPS)
If you access the service over plain HTTP, the API key is exposed to network interception.

**Recommendation:** always use HTTPS via a reverse proxy (Nginx/Caddy) or a tunnel solution, and restrict inbound access.

---

### 6) Temporary Files / Disk Usage
Some analysis flows create temporary directories (e.g., `tempfile.mkdtemp(...)`) that may not be removed automatically.

**Impact:** disk usage grows over time and can eventually break the server.

**Recommendation:** ensure temp directories are cleaned up (server-side) and monitor disk usage.

---

## Deployment Recommendations (Checklist)

**Minimum recommended controls:**

1. **Set a strong API key**
   - Use a long random secret (32+ bytes).

2. **Run behind HTTPS**
   - Terminate TLS at a reverse proxy/tunnel.

3. **Restrict network access**
   - Allow only trusted IPs/VPN subnets.
   - Consider `TrustedHostMiddleware` with `TRUSTED_HOSTS`.

4. **Avoid high concurrency**
   - Run one analysis at a time, or redesign to multi-process isolation.

5. **Sandbox dynamic execution**
   - If possible: execute code in an isolated container/VM (recommended).
   - Disable outbound network for the execution environment if not needed.

6. **Reduce information leakage**
   - Do not return full tracebacks to clients in production.

---

## Environment Variables

| Variable | Required | Example | Notes |
|---|---:|---|---|
| `YOUR_API_KEY` | Yes | `export YOUR_API_KEY="..."` | Primary API key for authentication. |
| `API_KEYS` | No | `export API_KEYS="k1,k2"` | Optional additional keys (comma-separated). |
| `PORT` | No | `export PORT=1234` | Server port. |
| `ALLOWED_ORIGINS` | No | `export ALLOWED_ORIGINS="https://your-ui.com"` | Avoid `*` in production. |
| `TRUSTED_HOSTS` | No | `export TRUSTED_HOSTS="yourdomain.com,localhost"` | Restricts Host header. |
| `MAX_REQUEST_SIZE` | No | `export MAX_REQUEST_SIZE=52428800` | Bytes; default ~50MB. |
| `TIMEOUT_KEEP_ALIVE` | No | `export TIMEOUT_KEEP_ALIVE=1800` | Keep-alive seconds. |

**Example (Linux/macOS):**
```bash
export YOUR_API_KEY="replace-with-a-strong-random-secret"
export PORT=8000
export ALLOWED_ORIGINS="https://your-frontend.example"
export TRUSTED_HOSTS="your-frontend.example,localhost,127.0.0.1"
python deploy.py
```

---

## API Usage

### Required Header
All requests must include:
- `X-API-Key: <your key>`

### Example: `curl`
```bash
curl -X POST "http://localhost:8000/deploy" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY_HERE" \
  -d '{"model_path":"gpt2"}'
```

### Example: Python (requests)
```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "YOUR_KEY_HERE"

headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# Submit an async analysis
resp = requests.post(
    f"{API_URL}/run_analysis",
    headers=headers,
    json={
        "feature": "cka",
        "model_reference_path": "gpt2",
        "model_path": "gpt2",
        "query": ["Hello world"],
        "device": "cuda",
        "batch_size": 1,
        "num_batches": 1,
        "max_length": 128,
        "analysis_code": None,
    },
)
task_id = resp.json()["task_id"]

# Poll status
status = requests.get(f"{API_URL}/task_status/{task_id}", headers=headers).json()
print(status)
```

---

## Known Limitations

- This server is best suited for **research/development** settings.
- Multi-user, high-concurrency usage may lead to:
  - incorrect results
  - crashes
  - cross-request interference
- Dynamic code execution is inherently dangerous and should be treated as privileged access.

---

## Disclaimer

This service enables privileged execution of code for research workflows. The operator is responsible for securing the environment, protecting API keys, and preventing unauthorized access or misuse.
