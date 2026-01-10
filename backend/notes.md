# Security Audit: `unlearning_deploy.py` Vulnerability Assessment

**⚠️ CRITICAL WARNING:** This code contains severe security vulnerabilities. It permits arbitrary remote code execution (RCE) without sandboxing or authentication. **Do not deploy this to a public network or any environment containing sensitive data.**

## 1. Critical Vulnerabilities

### A. Remote Code Execution (RCE)
- **Location:** `run_dynamic_code` endpoint and `execute_analysis_code` function.
- **Issue:** The code uses the Python built-in `exec()` function to run string input received directly from the client (`req.code`).
- **Risk:** This is the highest possible severity. An attacker can execute **any** Python command on the host server.
    - **Data Theft:** `import os; print(os.environ)` reveals API keys, database credentials, and system paths.
    - **File Access:** Attackers can read sensitive files (e.g., `/etc/passwd`, `~/.ssh/id_rsa`) or upload malicious files.
    - **System Control:** Attackers can open a reverse shell (`import socket...`) to take full control of the server.

### B. Lack of Isolation (No Sandboxing)
- **Issue:** The dynamic code runs in the same process and memory space as the web server.
- **Risk:** Even though `exec_globals` attempts to limit context, it does not prevent imports. A user can simply `import sys` or `import shutil` to bypass restrictions and destroy the file system or access network resources.

### C. Cross-User Data Leakage (Global State)
- **Location:** Global variables `GLOBAL_REFERENCE_MODEL`, `GLOBAL_UPDATED_MODEL`, and `TASKS`.
- **Issue:** The server is stateful and shares variables across all requests.
- **Risk:**
    - **Model Theft:** If User A loads a proprietary model, User B can write a script to access `model_ref` (which points to User A's model) and extract weights or architecture details.
    - **Task Snooping:** The `TASKS` dictionary is shared. If an attacker guesses or brute-forces a UUID, or simply iterates through memory objects via the RCE vulnerability, they can see other users' code, prompts, and analysis results.

### D. Information Disclosure via Tracebacks
- **Location:** `except Exception` blocks returning `traceback.format_exc()`.
- **Issue:** Detailed Python stack traces are returned to the API client upon error.
- **Risk:** This reveals the server's absolute file paths, installed library versions, and internal code structure, which aids attackers in crafting specific exploits.

### E. Missing Authentication
- **Issue:** The FastAPI app has no middleware for API keys, OAuth, or tokens.
- **Risk:** Anyone with network access to the port can execute code and control the server.

---

## 2. Privacy Impact Assessment

| Data Type | Risk Level | Description |
| :--- | :--- | :--- |
| **Model Weights** | 🔴 Critical | Proprietary models loaded into memory can be dumped by any user. |
| **User Code** | 🔴 Critical | Code sent for analysis is stored in the global `TASKS` dict and is accessible. |
| **Server Env Vars** | 🔴 Critical | API Keys (OpenAI, AWS, HF) stored in env vars are readable via RCE. |
| **File System** | 🔴 Critical | Local training data or config files can be read/exfiltrated. |

---

## 3. Recommendations for Remediation

1.  **Implement Sandboxing (Mandatory):**
    *   Never run `exec()` on the host machine.
    *   Use **Docker containers**, **Firecracker microVMs**, or **gVisor** to execute user code in an isolated environment.
    *   Destroy the container immediately after execution.

2.  **Add Authentication:**
    *   Implement `API Key` or `Bearer Token` verification for all endpoints.

3.  **Remove Global State:**
    *   Refactor the application to be stateless where possible.
    *   If models must be cached, use a session-based approach where User A cannot access User B's loaded objects.

4.  **Sanitize Error Outputs:**
    *   Log full tracebacks internally (to a file or monitoring system) but return only generic error messages (e.g., "Internal Execution Error") to the client.

5.  **Network Isolation:**
    *   Ensure the server running this code has no outbound internet access (unless strictly necessary) to prevent data exfiltration.
