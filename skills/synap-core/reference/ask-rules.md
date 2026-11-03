# What to ask the human, and how

**Never guess. If anything is unclear, stop and ask.** One question at a time. **Every question carries your recommendation** (what you suggest and why, from what you saw on the machine and in the project) **and a tip** (what the answer changes in the result). Say which answers are final and which can be changed later. Ask in plain words; do not use internal terms such as "edition" or "instance".

## The questions, in order

1. **Where should it live, and which install path?** (`choose-install-path.md`)
   Recommend: the clone path in a folder of its own such as `~/synap-core`, unless they only want it running and the image is public. Tip: the clone path builds from source (5 to 10 minutes) and lets them read the code; the image path is faster.
2. **Is something already installed?** Only if `preflight.py` found containers or volumes. Ask: keep the old data, or start fresh? Recommend keeping it unless they say it is a throwaway. Tip: a start under the same project name reuses the old memories, passwords and scoping mode. Deleting anything needs an explicit yes after you list what is lost.
3. **Consumer app or business app?** (`scoping-mode.md`) Ask who owns the data: the person, or an organisation.
   Recommend from the project if you can see one; otherwise recommend the consumer mode (`equals_customer`) as the simpler start. Tip: this cannot be changed after the first memory, and the verify step stores one. Skip it for a kept install: its mode is already fixed.
4. **Which model?** (`model-setup.md`) A provider and a model name.
   Recommend a hosted model they already have a key for. Tip: one model does every step, so a weak model weakens extraction; every message makes several model calls; a local Ollama model keeps data on the machine but is less accurate. If they have no preference, ask which provider key they already have.
5. **How will you get the key?** Never ask them to paste it. First ask: "was this agent started from the terminal where you exported the key?" If yes, `set_env.py --from-env`. If not, they type it into `<folder>/.env` themselves and tell you when it is saved. Tip: the key never enters this chat or any log.
6. **Protect the server with a token?** Recommend yes if anything other than their own backend on this machine will call it, otherwise optional. Tip: without a token anyone who can reach the port can read and write every memory; the port is bound to the machine only.
7. **Will you send images?** Recommend: only if yes, set `SYNAP_LLM_MODEL_VISION`. Tip: otherwise the "No vision model" warning is harmless.
8. **Tell me about your agent** (during the first start; `use-case-questions.md`). Three short rounds: what it does and who uses it; what a user says and what it must remember or never store; anything else. Read their project first if there is one and propose answers to correct. Tip: the answers become `use-case.md`, and the better it is, the better the memory config MACA designs.
9. **Design the memory config now?** "I can have Synap design a memory config for your agent from that file. It needs a Synap Cloud account with credits (https://synap.maximem.ai) and takes about 20 minutes, or you can stay on the starter config." Recommend yes if they have an account. Tip: finish this before the verify step, because the scoping mode is final once a memory exists. Do not start MACA unless they say yes.

Do not ask what you can check yourself (the OS, Docker, free disk, ports, existing volumes): run `preflight.py`.

## Confirm before you do these

- Start the stack for the first time (say the time and the 2.3 GiB download).
- Anything that deletes: `down -v`, `docker volume rm`, deleting `originals/`, the Synap Core folder, `.env`, `config/memory*.yml` or `~/.synap`, changing the scoping mode after memories exist.
- Sending the use-case file to Synap Cloud (show the exact file first).
- Switching `config/memory.yml` to a generated config (show the diff first).

## Hard rules

- **Never ask the human to paste a key, token or password into chat, and never print one.** Use `scripts/set_env.py` and `scripts/check_env.py`, which show names and lengths only. **Never open, read, grep or print `.env`, or the output of `docker compose config`, with any tool** (not `cat`, not the Read tool, not an Edit that needs a Read first): that would put the key in your context. Write `.env` only through `set_env.py`. Never `docker inspect` an environment or echo a variable that holds a key.
- **Never run `git commit`, `git add -A`, `git push`, `git stash` or an amend.** Leave changes uncommitted and new files untracked for the human to review. `.env` is already ignored by the repository.
- Touch only Synap Core's own Compose project (`synap-core`, or the name the human chose). Never `docker system prune` or `docker volume prune`; never stop or remove containers that are not Synap Core's.
- Do not change the Compose file's published address or ports. Do not edit `synap/` (the engine).
- If something fails: read `troubleshooting.md`; if it is not there, collect the diagnostics it lists and stop and ask. Never try a destructive fix to make an error go away.
- Never claim success the scripts did not print. Quote `preflight.py`, `check_env.py`, `wait_healthy.py` and `verify_core.py` output unedited, including every `[WARN]`.
- Be honest about what Core does not have (`mental-model.md`); do not offer Cloud-only features.
