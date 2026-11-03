# Connect the SDK to the running server

Synap Core uses the same published SDKs as Synap Cloud. Point them at the server and leave the API key out: a local server needs no API key and no instance id.

```bash
# Python 3.11+ (0.6.13 or newer; older versions have no sdk.maca)
pip install maximem-synap
# JavaScript / TypeScript, Node 20+
npm install @maximem/synap-js-sdk

export SYNAP_BASE_URL=http://127.0.0.1:8000
```

**Unset `SYNAP_API_KEY` in the shell first.** A shell that still has the key from a Synap Cloud project makes the SDK behave as a Cloud client even when `SYNAP_BASE_URL` points at this server: local mode is only on when there is no key, so the SDK sends the key, reports telemetry and lets `listen()` open a stream like Cloud (without a key, Core refuses `listen()`). Check by name, never by value, and remove it for this shell:

```bash
env | grep -o '^SYNAP_API_KEY'     # prints the name if it is set
unset SYNAP_API_KEY
```

Do the same in the shell, `.env` file or process manager that starts your agent, and pass no `api_key=` argument.

If the server has a token (`SYNAP_API_TOKEN` in its `.env`), set the same variable in the SDK's environment. Both SDKs read `SYNAP_BASE_URL` and `SYNAP_API_TOKEN` from the environment. The server's identity is fixed: client `cli_core_default`, instance `inst_core_default`.

## Store a memory and read it back (Python)

```python
import asyncio
from maximem_synap import MaximemSynapSDK

async def main():
    sdk = MaximemSynapSDK()
    await sdk.initialize()
    try:
        result = await sdk.memories.create(
            document="User: I prefer dark mode.\nAssistant: Noted!",
            document_type="ai-chat-conversation",
            user_id="user_alice",              # equals_customer: a user id only
        )                                      # strict: also pass customer_id="acme"
        await sdk.memories.wait_for_completion(result.ingestion_id)

        context = await sdk.user.context.fetch(user_id="user_alice", search_query=["preferences"])
        for p in context.preferences:
            print(p.content)                   # "The user prefers dark mode."
    finally:
        await sdk.shutdown()

asyncio.run(main())
```

JavaScript: `import { SynapClient } from '@maximem/synap-js-sdk'; const sdk = new SynapClient(); await sdk.initialize(); ... sdk.memories.create({...}); sdk.memories.wait_for_completion(id); sdk.user.context.fetch({...})`. See `examples/`.

Synap rewrites what the user said into a short statement about them, so the sentence that comes back differs from the one sent. The first memory can take up to a minute while models load.

## Things the SDK does that surprise people

- **A five-minute cache.** The SDK keeps answers it already fetched in `~/.synap/cli_core_default/inst_core_default`. Right after deleting or changing a memory, or wiping the server, the same query can show the old answer until the five minutes end. `sdk.cache.clear()` (or deleting that folder) shows the server's current answer. This is by design.
- **Retries.** A failed call is retried up to three times with a short wait ("Request failed (attempt 1/3)"). Right after a restart, "Connection closed by peer" means the server is still loading: wait for healthy.
- **Cloud-only calls** raise an error whose message is "... not available in the open-source server": `credits.*`, `data.delete` (everything for a user), `instance.listen` and the live stream. Core has no live stream: use the explicit path (write with `memories.create` or `conversation.record_messages_batch`, read with `user.context.fetch` / `conversation.context.fetch`).
- **A short `document`** (under 100 characters; `email` under 50) is skipped, and the ingestion finishes as `completed` with `memories_created` 0 and an `error_message` that says why. Send short text as `ai-chat-conversation`.
- **Compaction** runs on its own for conversations; to trigger one, `await sdk.conversation.context.compact(conversation_id, force=True)`, then poll `get_compaction_status` and read `get_compacted`.

## Framework integrations

LangChain, LangGraph, LlamaIndex, CrewAI and the rest are the same packages Synap Cloud uses (`maximem-synap-<framework>` on PyPI, `@maximem/synap-<framework>` on npm), listed with links in the repository README. Point them at Core with `SYNAP_BASE_URL` and no key. Set `SYNAP_AUTO_LISTEN=false` in the agent's environment. Integrations that report turns and tool calls try to open a live gRPC stream the first time they report something, and Core has no stream. Without the setting each one logs a warning once, `synap: could not open the gRPC stream`, and tries again about once a minute. Turns still go over REST, so nothing breaks, but the log is noisy. With it set, the attempt is skipped. Wiring one into an agent is covered by the separate `synap` skill: `agent-framework-handoff.md`.
