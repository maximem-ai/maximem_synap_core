# Use-Case Description

## Agent Objective

Remy is a personal chatbot. It chats with one person at a time about their day, their plans and the things they care about, and it remembers what the person has told it so that the next conversation starts from what it already knows, not from nothing. It is a companion and a planner, not a search engine: its value is that it knows the person.

## Target Users

Individual adults chatting on their own, mostly non-technical. Each person is their own customer: there is no company or team above them. People come back several times a week for short chats, and some return after a few days to pick up something said earlier.

## Interaction Pattern

Text chat in a terminal or a simple app. A session is 5 to 20 messages, with a few seconds to a minute between messages. Follow-ups to an earlier conversation are common ("how did the trip go?").

## Task Examples

- **User**: "My name is Sam and I'm vegetarian."
  **Agent**: Remembers the name and the diet as standing facts, and stops suggesting meat.
- **User**: "I'm planning a trip to Lisbon in March, five days, on a budget."
  **Agent**: Remembers the destination, the month, the length and the budget, and uses them when asked what to pack or where to eat.
- **User**: "I started learning the guitar, I practise in the evenings."
  **Agent**: Remembers the hobby and the routine, and asks about progress later.
- **User**: "Actually I'm not vegetarian any more, I eat fish now."
  **Agent**: Replaces the old diet with the new one instead of keeping both, and keeps the change in the record.
- **User**: "My sister Priya's birthday is on 14 March."
  **Agent**: Remembers the person, how she is related, and the date, attached to the person who said it.

## Behavioral Guidelines

- Treat the latest statement as the truth when it contradicts an older one.
- Never store passwords, card numbers, government ID numbers or one-time codes.
- Do not guess personal facts the person has not said.
- Keep what the person says about other people (family, friends, colleagues) attached to the person who said it.
- Do not bring up sensitive details (health, money, relationships) unless the person raised them in this conversation or asked about them.

## Role Descriptions

- **Client**: the company that runs the chatbot.
- **Customer**: the person chatting. Each person is their own customer.
- **User**: the same person.

## Compliance and Data Sensitivity

Diet, health, relationships, locations and travel plans can come up and are personal. People can ask to have anything forgotten, and it must be removable by memory. No special regulation is named; assume the data is personal and treat it with care.

## Memory Priorities

Remember first: the person's name, standing preferences (diet, tone, schedule), goals and plans with dates, important dates, the people in their life and how they relate to them, and routines or hobbies. Remember with lower priority: one-off remarks and small talk.

## Additional Context

English only. A small number of people per instance to begin with. Dates matter: record absolute dates when the person gives them ("14 March", "next March" with the year when it is clear).
