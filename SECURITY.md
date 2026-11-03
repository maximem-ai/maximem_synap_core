# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: go to the **Security** tab of this repository and choose **Report a vulnerability**. You can also email [security@maximem.ai](mailto:security@maximem.ai). We acknowledge reports within 72 hours and keep you updated until the issue is fixed.

## Scope

- The Synap Core server in `synap/`
- The Docker setup in `deploy/` and the `Dockerfile`
- The settings and sample configs in `config/` and `.env.example`

Issues in the hosted Synap Cloud service, the SDKs and the framework integrations can be reported the same way. They live in other repositories, and we will route the report.

## Running Core safely

Synap Core is built to sit behind your own backend, not to face the internet.

- **It has no login by default.** Anyone who can reach the port can read, write and delete every memory. The API listens on `127.0.0.1:8000` only. Keep it that way, or put a reverse proxy with TLS in front of it.
- **Set `SYNAP_API_TOKEN`** in `.env` to require a bearer token on every request. The server logs a warning at start while it is unset.
- **No database port is published**, and the database, Redis and graph passwords are generated on the first start and kept in a Docker volume. There are no default passwords.
- **What is stored comes back into your agent's prompt.** A memory can carry instructions (text a user pasted, or something planted by anyone who can write). Treat recalled memories as untrusted data in your prompts, and do not store text from sources you do not trust.
- **Every write runs your model provider.** Anyone who can reach the port can run up your model bill by sending text to ingest. The token and the local-only port are the protection; set a spending limit with your provider as well.
- **Keep `.env` private** (`chmod 600 .env`) and out of version control. It holds your model provider key.
- **Two optional keys** can sit in `.env` besides your model provider key. The **Synap Cloud key** for the MACA plugin is sent to Synap Cloud and nowhere else, and the server never returns it or writes it to a log. The **Jev key** (`JEV_API_KEY`) is used only for forget by description, and it sends the text of the candidate memories to Jev (`api.typesafe.ai`). Leave either one unset, or remove it, when you are not using that feature.
- **Do not call the server from a browser or a mobile app.** Call it from your own backend.

If a key is exposed, revoke it with its provider and create a new one.

See [Running in production](https://docs.maximem.ai/core/production) for the full list.
