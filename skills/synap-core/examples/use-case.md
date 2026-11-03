# Use-Case Description

## Agent Objective

Atlas is a personal assistant in a consumer mobile app. It helps one person at a time plan their week, keep up with their habits and goals, and remember the people and preferences that matter to them, so they never have to repeat themselves. It answers in a few sentences, in a warm and direct tone.

## Target Users

Individual adults using the app on their own account, mostly non-technical. Each person is their own customer: there is no company or team above them. People come back several times a week, in short sessions, and expect Atlas to remember what they said last month.

## Interaction Pattern

Text chat in the app. A session is 3 to 15 messages, with 10 to 60 seconds between messages. Some people return after days or weeks with a follow-up to something said earlier.

## Task Examples

- **User**: "Remind me that Priya's birthday is on 14 March, she loves orchids."
  **Agent**: Remembers the person (Priya), the date, and the preference, and brings it up a week before.
- **User**: "I'm vegetarian now, no fish either. Can you plan dinners for this week?"
  **Agent**: Remembers the diet as a standing constraint, replaces an earlier "eats fish on Fridays" note, and plans accordingly.
- **User**: "I'm training for a 10k in May and I run Tuesday and Saturday mornings."
  **Agent**: Remembers the goal, the date and the routine, and adjusts later suggestions.
- **User**: "Actually I moved the long run to Sunday."
  **Agent**: Updates the routine instead of keeping both.

## Behavioral Guidelines

- Treat the latest statement as the truth when it contradicts an older one, and keep the change in the record.
- Never store payment card numbers, passwords, government ID numbers or one-time codes.
- Do not guess personal facts the person has not said.
- Keep what the person said about other people (family, friends, colleagues) attached to the person who said it.

## Role Descriptions

- **Client**: the company that runs the Atlas app.
- **Customer**: the person using the app. Each person is their own customer.
- **User**: the same person.

## Compliance and Data Sensitivity

Health and diet details, relationships and locations can come up and are personal. People can ask to have anything forgotten, and it must be removable by memory. No special regulation is named; assume general consumer privacy expectations.

## Memory Priorities

Remember first: standing preferences (diet, schedule, tone), goals and deadlines, important dates, people and how the person relates to them, and routines. Remember with lower priority: one-off plans for the current week and moods. Do not track small talk, greetings or the assistant's own suggestions that the person did not accept.

## Additional Context

English only. Mobile app, users spread across time zones, so dates matter: record absolute dates when the person gives them. Expect a few thousand people per instance and a handful of memories per session.
