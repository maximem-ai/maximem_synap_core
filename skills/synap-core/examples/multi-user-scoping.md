# Scoping in practice

Two users never see each other's memories. Check it on your own install:

```python
# equals_customer (consumer): a user id only
await sdk.memories.create(document="User: I am vegetarian.\nAssistant: Noted.", document_type="ai-chat-conversation", user_id="ana")
await sdk.memories.create(document="User: My sister Priya loves orchids.\nAssistant: Noted.", document_type="ai-chat-conversation", user_id="ben")
await sdk.user.context.fetch(user_id="ana", search_query=["Priya"])   # nothing about Priya
await sdk.user.context.fetch(user_id="ben", search_query=["diet"])    # nothing about a vegetarian diet
```

```python
# strict (business): a customer id AND a user id on every call
await sdk.memories.create(document="...", document_type="ai-chat-conversation", user_id="ana", customer_id="acme")
await sdk.user.context.fetch(user_id="ana", customer_id="acme", search_query=["..."])
# a fetch for the customer alone returns only what was stored at the customer level
```

| Mistake | What Core answers |
|---|---|
| a `customer_id` on an `equals_customer` install | rejected, naming the customer id |
| a user id without a customer id on a `strict` install | an error that says a customer id is needed |
| an unknown user | an empty result, never someone else's |

`python3 scripts/verify_core.py` runs the isolation check for you, with throwaway users, and deletes what it wrote.
