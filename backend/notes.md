# Remote Deployment Safety Notes

The unlearning deployment service is a powerful research tool, not a
general-purpose public API. Authenticated analysis endpoints execute Python
code supplied by the client, and model loading may execute repository code via
`trust_remote_code=True`. Anyone with an API key should therefore be treated as
having code-execution access to the server account.

## Before starting the service

- Use a dedicated container, virtual machine, or non-privileged account. Do not
  run the service on a workstation or server that holds unrelated credentials.
- Give the service access only to the model and output directories it needs.
  Apply CPU, memory, GPU, process, and request limits appropriate to the host.
- Set a long, randomly generated `YOUR_API_KEY`; never commit it, put it in a
  URL, or share it in screenshots and logs.
- Restrict `ALLOWED_ORIGINS` to the frontend origin and set `TRUSTED_HOSTS` to
  the expected deployment hostname. Their defaults are intentionally convenient
  for development and are not suitable for an exposed service.
- Keep port `1234` blocked from direct Internet access. A Cloudflare quick
  tunnel URL is publicly reachable even though requests still require the API
  key, so share both values only with trusted operators.
- Load only models and analysis code from sources you trust. The API key does
  not sandbox submitted code or code shipped with a model.

Example environment configuration:

```bash
export YOUR_API_KEY="$(openssl rand -hex 32)"
export ALLOWED_ORIGINS="https://your-frontend.example"
export TRUSTED_HOSTS="your-tunnel.trycloudflare.com"
export MAX_REQUEST_SIZE="10485760"
python backend/unlearning_deploy.py
```

## After the session

Stop both `cloudflared` and the deployment service, rotate any key that may have
been disclosed, and review service logs before deleting them. Models and task
results are retained in process memory only, so stopping the process clears that
state; files created by submitted code are outside that guarantee and must be
reviewed separately.
