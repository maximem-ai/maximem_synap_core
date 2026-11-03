# After setup: wiring memory into an agent

This skill sets up the **server**. It stops there unless the human asks for more. Wiring memory into their agent framework is a different job, covered by the `synap` skill in the SDK repository:

```bash
npx skills add maximem-ai/maximem_synap_sdk --skill synap -y
```

(or https://github.com/maximem-ai/maximem_synap_sdk/tree/main/skills/synap). That skill was written for Synap **Cloud**. When you use it with Core, these differences apply, and they win over what it says about Cloud:

| In the Cloud skill | With Synap Core |
|---|---|
| sign in with a one-time code or `auth login`, create an instance with a use-case file, wait 15 to 20 minutes for the build | skip all of it: the server is already running, with one fixed instance |
| create an API key and write `SYNAP_API_KEY` and `SYNAP_INSTANCE_ID` | none. Set `SYNAP_BASE_URL=http://127.0.0.1:8000`, and `SYNAP_API_TOKEN` only if the server has a token |
| an integration opens the live stream on first use | set `SYNAP_AUTO_LISTEN=false` in the agent's environment. Core has no stream, so without it the integration logs "could not open the gRPC stream" once and retries about once a minute (turns still go over REST) |
| the implicit path: `listen()`, `send_message`, `record_*` over the live stream | **not available** (`listen()` raises "Not available in the open-source server"). Use the explicit path |
| B2C or B2B chosen when the instance is created | the scoping mode is already fixed by `config/memory.yml` (`scoping-mode.md`); read it from `GET /api/v1/auth/whoami` (`user_context_isolation`) |
| `verify_synap.py`, `cleanup_test_data.py` | use `scripts/verify_core.py` from this skill |
| dashboard, usage, credits, instance visibility | none of these exist in Core |

The explicit path in one line each: **write** each finished turn or document with `sdk.memories.create(document=..., document_type="ai-chat-conversation", user_id=...)` (or `conversation.record_messages_batch`); **read** before each model call with `sdk.user.context.fetch(user_id=..., search_query=[...])` and put the result into the prompt. Memory informs; the app's own rules decide. Never store payment card numbers, passwords or one-time codes.

If the human uses a coding agent that already has the `synap` skill, tell it to follow the table above. The packages for each framework are listed with links in the repository README.
