# Typed tools and transactional safety

## Tool surface

| Tool | Capability | State effect |
|---|---|---|
| `search_catalog` | Retrieve product candidates | Read |
| `get_product` | Read one canonical product | Read |
| `compare_products` | Compare canonical products | Read |
| `get_policy` | Retrieve one policy category | Read |
| `get_order` | Read an authenticated order | Read |
| `check_return_eligibility` | Evaluate return rules | Read |
| `create_return_request` | Create an idempotent return request | Write |
| `escalate_to_human` | Record a handoff | Controlled action |

## Guardrail chain

High-risk return operations require all of the following:

1. the session identity and six-digit verification code match;
2. the requested order belongs to that user;
3. the order satisfies the return policy;
4. the user has explicitly confirmed the write;
5. the requested write is valid and idempotent.

The harness injects session identity independently of model-provided arguments, so the policy cannot operate on behalf of another user by changing `user_id`.

## Safety interpretation

Agent v2 recorded a 5% forbidden-tool attempt rate and a 0% illegal-state-change rate. This means the execution layer protected the database even when the policy selected an invalid action; it does not mean policy compliance was perfect.
