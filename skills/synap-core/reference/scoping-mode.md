# Choose the scoping mode (before anything is stored)

How do the app's users relate to customers? The human decides; you recommend from what the project shows.

| Mode | For | What every request carries |
|---|---|---|
| `equals_customer` | a consumer app: each user is their own customer | a user id only. A customer id is refused. |
| `strict` | a business app: users sit under a customer (an organisation) | a customer id AND a user id. A user id alone is an error. |

The repository ships `config/memory.yml` set to `equals_customer`, and `config/samples/business-app.yml` set to `strict`.

## It cannot be changed once memories exist

Memories are filed under the ids of the mode they were stored in; switching would leave them where the new mode does not look. So:

- Before the first memory: change your mind freely. Edit `user_context_isolation` in `config/memory.yml` and run `docker compose -f deploy/docker-compose.yml up -d`.
- After any memory exists: the server refuses to start with a different mode, and a live edit is ignored with an error in the log. To change, start over (`uninstall-and-reset.md`), which deletes the stored memories. Never do that without an explicit yes.

**If the human wants a business app, make the change before the first start:**

```bash
cp config/samples/business-app.yml config/memory.yml       # clone path
# image path: download that sample first (the command is in choose-install-path.md), then the same cp
```

A kept install (earlier volumes with memories) already has its mode: do not ask, read it with `GET http://127.0.0.1:8000/api/v1/auth/whoami` once it is up, or from `config/memory.yml`.

**The verify step counts as the first memory.** `verify_core.py` stores and then deletes test memories, and a delete leaves some residue (a conversation or graph entity), so treat the mode as final once you have run it. Decide before you verify.

## Wrong requests fail loudly

A customer id on an `equals_customer` install is rejected with a message that says what the server expected (HTTP 400, "customer_id not accepted"). A missing customer id on a `strict` install is an error too. Nothing quietly returns empty results.

## What Core places where

A fact is stored at the level of the request: the ids sent decide where it lives. A fact that is really about a whole customer stays with the user who said it (Synap Cloud places facts automatically; Core does not). A fetch for a customer returns only what was stored at the customer level (sent with a customer id and no user id); it does not add up that customer's users. To read a user's memories, fetch for that user.

Core also refuses `maca_suggests` as a scoping mode, and the `scope_path` request field (Core has no custom levels).

## Words to use with the human

"Each of your users is their own account and nobody sits above them" is `equals_customer`. "Your users belong to a company or team that owns the data" is `strict`. If they are unsure, ask who owns the data: the person, or an organisation.
