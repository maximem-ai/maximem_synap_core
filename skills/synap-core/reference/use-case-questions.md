# Learning about the agent: the questions that become `use-case.md`

Do this in step 7, while the stack starts and downloads its models. The answers become `<folder>/use-case.md`, the file the MACA plugin sends to Synap Cloud so it can design a memory config for THIS agent. The better the file, the more the config remembers the right things and the less it keeps that it should not. Without a file the human stays on the starter config, which works but is generic.

The rules in `ask-rules.md` apply: every question carries your **recommendation** and a **tip** (what the answer changes in the result). Ask a short group of questions at a time, three rounds in all. Never ask for a key, a password or real customer data, and never put one in the file.

## Look first, then ask

If there is a project in sight (a README, a system prompt, a list of tools, a folder of prompts), read it first. Draft an answer to each question from what is there, then ask the human to correct it. "Is this right?" is faster for them than a blank question, and it is more accurate than guessing. If there is nothing to read, ask.

Never invent. A fact the human did not say, and you could not read, goes in the file as `not stated`.

## Round 1: what the agent is

| # | Ask | Goes into | Recommend, and tip |
|---|---|---|---|
| 1 | What does your agent do, in one or two sentences? | **Agent Objective** | Recommend their own words. Tip: this frames every choice the config makes; a vague answer ("a helpful assistant") gives a vague config. |
| 2 | Who talks to it, and how often does the same person come back? | **Target Users** | Do not ask whether it is a consumer or business app: step 4 already decided that, so state it ("you chose consumer: each person is their own customer") and ask only about the people. Tip: how often people return decides how much is worth remembering. |
| 3 | How do they talk to it, and how long is a session? | **Interaction Pattern** | Recommend a rough shape ("text chat, 3 to 15 messages, people return after days"). Tip: session length sets how much text each memory is built from. |

## Round 2: what to remember

| # | Ask | Goes into | Recommend, and tip |
|---|---|---|---|
| 4 | Give me three to five real things a user might say, and what the agent should remember from each. | **Task Examples** | Offer three you drafted from rounds 1 and 2 ("I'm vegetarian now, no fish either" and so on) and let them keep, change or replace them. Tip: this is the section that matters most; it tells the config which kinds of information are worth keeping. |
| 5 | What must it never forget about a person? | **Memory Priorities** | Recommend two or three things that follow from the examples (diet, goals, important dates). Tip: the config ranks these first. |
| 6 | What must it never store? | **Behavioral Guidelines** and **Compliance and Data Sensitivity** | Recommend the usual: passwords, card numbers, government ID numbers, one-time codes. Ask whether people can ask to be forgotten. Tip: this decides what the config filters out and where its boundaries are. |

## Round 3: the rest (skip it if there is nothing)

| # | Ask | Goes into |
|---|---|---|
| 7 | Anything else the config should know: languages, time zones, how many people, anything unusual? | **Additional Context** |

**Role Descriptions** are not a question. Derive them from step 4 and confirm in one line:

- Consumer (`equals_customer`): **Client** is the company that runs the app; **Customer** is the person using it; **User** is the same person.
- Business (`strict`): **Client** is the company that runs the app; **Customer** is the organisation using it; **User** is a person at that organisation.

## Write the file

Save `<folder>/use-case.md`, UTF-8, under 256 KB, with exactly these nine sections in this order:

```
# Use-Case Description

## Agent Objective
## Target Users
## Interaction Pattern
## Task Examples
## Behavioral Guidelines
## Role Descriptions
## Compliance and Data Sensitivity
## Memory Priorities
## Additional Context
```

- **Task Examples** are pairs, one per example, in this shape: `- **User**: "what they say"` then, on the next line, `  **Agent**: what it does and what it remembers`.
- Plain sentences, in the human's words where you can. No headings of your own, no tables, no code.
- Write `not stated` in a section the human gave you nothing for. Do not pad it.
- No key, token, password, real name or real customer data. Examples use made-up people.

Then run `python3 <skill>/scripts/check_use_case.py` (it checks the three required sections, the example count, secrets and size, and names every problem). Show the human the WHOLE file, not a summary, and get a yes. Offer to change anything. The file does not leave the machine until step 8 and their yes.

## Two finished examples

- `examples/use-case.md`: a consumer assistant for planning a week.
- `examples/use-case-personal-chatbot.md`: a personal chatbot that remembers the person it talks to.

## What to say when they have no agent yet

Say so plainly and offer the starter config: it works for any app and can be tuned by hand. Write a `use-case.md` later, when there is something to describe. Do not make one up to have a file.
