"""Store a memory in a local Synap Core server and read it back. No API key, no instance id.

    pip install maximem-synap            # 0.6.13 or newer
    export SYNAP_BASE_URL=http://127.0.0.1:8000
    python python-minimal.py

If the server has a token, also export SYNAP_API_TOKEN. On a business (strict) install set
SYNAP_CUSTOMER_ID=acme so every call carries a customer id; on a consumer install leave it unset
(a customer id is refused there).
"""
import asyncio
import os

from maximem_synap import MaximemSynapSDK

CUSTOMER = {"customer_id": os.environ["SYNAP_CUSTOMER_ID"]} if os.environ.get("SYNAP_CUSTOMER_ID") else {}


async def main() -> None:
    sdk = MaximemSynapSDK()
    await sdk.initialize()
    try:
        result = await sdk.memories.create(
            document="User: I prefer dark mode.\nAssistant: Noted!",
            document_type="ai-chat-conversation",
            user_id="user_alice",
            **CUSTOMER,
        )
        await sdk.memories.wait_for_completion(result.ingestion_id)

        context = await sdk.user.context.fetch(user_id="user_alice", search_query=["preferences"], **CUSTOMER)
        for p in context.preferences:
            print(p.content)  # "The user prefers dark mode."
    finally:
        await sdk.shutdown()


asyncio.run(main())
