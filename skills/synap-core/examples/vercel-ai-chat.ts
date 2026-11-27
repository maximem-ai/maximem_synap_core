// Chat loop with the Vercel AI SDK and Synap Core.
//
// Fetches memory context for the user before each turn, injects it into the
// system prompt, and stores the completed conversation turn into Synap Core after.
//
// Setup:
//   npm install ai @ai-sdk/openai @maximem/synap-js-sdk
//   (Node 20+; the project must be ESM: "type": "module" in package.json, or name this file .mts)
//
// Run from the repository root:
//   unset SYNAP_API_KEY
//   export SYNAP_BASE_URL=http://127.0.0.1:8000
//   export OPENAI_API_KEY=sk-...
//   npx tsx skills/synap-core/examples/vercel-ai-chat.ts
//
// If the server requires an auth token, also set SYNAP_API_TOKEN.
// On a business (strict) install set SYNAP_CUSTOMER_ID=acme; on a consumer install leave it unset.
import { generateText, CoreMessage } from 'ai';
import { openai } from '@ai-sdk/openai';
import { SynapClient } from '@maximem/synap-js-sdk';

// A leftover Cloud API key makes the SDK act as a Cloud client: clear it.
delete process.env.SYNAP_API_KEY;

const customer = process.env.SYNAP_CUSTOMER_ID ? { customer_id: process.env.SYNAP_CUSTOMER_ID } : {};

async function chatWithMemory(
  sdk: SynapClient,
  userId: string,
  userMessage: string,
  history: CoreMessage[],
): Promise<string> {
  // 1. Fetch relevant memories for this user prior to answering
  const context = await sdk.user.context.fetch({
    user_id: userId,
    search_query: [userMessage],
    ...customer,
  });

  const memoryLines: string[] = [];
  for (const pref of context.preferences ?? []) {
    memoryLines.push(`- Preference: ${pref.content}`);
  }
  for (const fact of context.facts ?? []) {
    memoryLines.push(`- Fact: ${fact.content}`);
  }
  for (const ep of context.episodes ?? []) {
    memoryLines.push(`- Note: ${ep.content}`);
  }

  const memoryContext =
    memoryLines.length > 0
      ? `Relevant memories for this user:\n${memoryLines.join('\n')}`
      : 'No previous memories found for this user.';

  // 2. Generate response using Vercel AI SDK with injected memory context
  const { text: reply } = await generateText({
    model: openai(process.env.OPENAI_MODEL ?? 'gpt-4o-mini'),
    system: `You are a helpful assistant. Use the following context when answering if relevant:\n${memoryContext}`,
    messages: [...history, { role: 'user', content: userMessage }],
  });

  // 3. Store the turn in Synap Core after the reply is generated
  const turnDocument = `User: ${userMessage}\nAssistant: ${reply}`;
  const ingestion = await sdk.memories.create({
    document: turnDocument,
    document_type: 'ai-chat-conversation',
    user_id: userId,
    ...customer,
  });
  await sdk.memories.wait_for_completion(ingestion.ingestion_id);

  return reply;
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    console.error('OPENAI_API_KEY is not set. Please export OPENAI_API_KEY=sk-... before running.');
    process.exit(1);
  }

  const sdk = new SynapClient();
  await sdk.initialize();

  const userId = 'user_alice';
  const history: CoreMessage[] = [];

  try {
    // Turn 1: The user introduces their preferences
    const turn1Message = 'Hi! I prefer dark mode and concise TypeScript answers.';
    console.log(`\nUser: ${turn1Message}`);
    const reply1 = await chatWithMemory(sdk, userId, turn1Message, history);
    console.log(`Assistant: ${reply1}`);

    history.push({ role: 'user', content: turn1Message });
    history.push({ role: 'assistant', content: reply1 });

    // Turn 2: Follow-up question relying on memory recall from Synap Core
    const turn2Message = 'Can you suggest an IDE theme and a project template?';
    console.log(`\nUser: ${turn2Message}`);
    const reply2 = await chatWithMemory(sdk, userId, turn2Message, history);
    console.log(`Assistant: ${reply2}`);
  } finally {
    await sdk.shutdown();
  }
}

main().catch((err) => {
  console.error('Execution failed:', err);
  process.exit(1);
});
