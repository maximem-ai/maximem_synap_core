# Where the facts live

Everything this skill needs is in this folder. If you need more, in this order:

| For | Look at |
|---|---|
| the install flow, settings and what Core is | the repository README, `.env.example` (every setting is commented), `deploy/docker-compose.yml` |
| the server's own behaviour | `GET http://127.0.0.1:8000/browser/api/config`, `docker compose ... logs synap`, the memory browser |
| tests of an install | `e2e/README.md` in the repository: a 19-check suite that starts its own stack |
| reporting a problem | `CONTRIBUTING.md`, the issue form, `SECURITY.md` for vulnerabilities (privately) |
| SDK reference | https://docs.maximem.ai/sdk-reference/overview |
| the MACA use-case template | https://docs.maximem.ai/getting-started/use-case |
| wiring an agent framework | the `synap` skill (`agent-framework-handoff.md`), https://docs.maximem.ai/integrations |

The Synap Core section of the docs site (`docs.maximem.ai/core/...`: quickstart, configuration, local models, scoping, forget, images, MACA plugin, production, cost notes, Core vs Cloud) may not be published yet. Do not tell the human to open it unless a request to it returns a page; the files in this folder say the same things. Add `.md` to a docs.maximem.ai URL for clean text.
