// Store a memory in a local Synap Core server and read it back. No API key, no instance id.
//
//   npm install @maximem/synap-js-sdk        (Node 20+; the project must be ESM: "type": "module" in package.json, or name this file .mts)
//   export SYNAP_BASE_URL=http://127.0.0.1:8000
//   npx tsx typescript-minimal.ts
//
// If the server has a token, also export SYNAP_API_TOKEN. On a business (strict) install set
// SYNAP_CUSTOMER_ID=acme; on a consumer install leave it unset (a customer id is refused there).
import { SynapClient } from '@maximem/synap-js-sdk';

const customer = process.env.SYNAP_CUSTOMER_ID ? { customer_id: process.env.SYNAP_CUSTOMER_ID } : {};

const sdk = new SynapClient();
await sdk.initialize();
try {
  const result = await sdk.memories.create({
    document: 'User: I prefer dark mode.\nAssistant: Noted!',
    document_type: 'ai-chat-conversation',
    user_id: 'user_alice',
    ...customer,
  });
  await sdk.memories.wait_for_completion(result.ingestion_id);

  const context = await sdk.user.context.fetch({ user_id: 'user_alice', search_query: ['preferences'], ...customer });
  for (const p of context.preferences ?? []) console.log(p.content); // "The user prefers dark mode."
} finally {
  await sdk.shutdown();
}
