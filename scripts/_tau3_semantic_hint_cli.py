"""Launch tau2 with task-specific semantic plans that contain no entity IDs."""

from __future__ import annotations

import os

from tau2.agent.llm_agent import LLMGTAgent
from tau2.evaluator import evaluator_nl_assertions
from tau2.registry import registry


GLOBAL_PROCESS_RULES = """
The private plan is guidance about the intended reasoning process, not evidence.
Never reveal or act on a private-plan fact until the user has disclosed it or a
tool response has established it. Discover every user ID, order ID, product ID,
item ID, payment method and state through the conversation and read tools.

Authenticate before accessing account information. Treat write tools as final
actions, never as probes. Before every write, summarize the exact target,
arguments and consequences and obtain an explicit user confirmation. If a tool
accepts only enumerated values, do not silently map the user's wording to an
enum: ask the user to choose or explicitly confirm the allowed value. Never
claim that a refund, balance change, status change or amount exists unless a
previous tool response supports it. A successful return request is not proof
that money has already been refunded. Follow the Retail policy even if the
private plan omits a prerequisite.
""".strip()


SEMANTIC_PLANS = {
    "14": """
Identify the orders and ask the user which products they consider gaming
related. Return only the keyboard and mouse the user confirms; do not infer that
an action camera, backpack or other item is gaming related. Use each order's
original payment method and confirm both returns together before writing.
""",
    "20": """
Inspect every relevant pending order and all variants for each requested item.
For every item, compare available variant prices and select the highest-priced
valid variant; preserve the shoe size even if other options change. Collect all
changes for an order before the single modify-items call. Use the gift card only
if it can cover the difference; otherwise ask about the user's fallback.
""",
    "29": """
For the skateboard, find shorter bamboo variants, show every valid option and
price, and let the user select the highest-priced one. For the garden hose,
discover the desired variant by inspecting the user's pending order rather than
guessing or exposing a hidden ID. Confirm all exchange details before writing.
""",
    "30": """
Find and report the tablet tracking number. Check whether the exact replacement
is available; if not and the user chooses a return, confirm the return details.
If the user then asks to cancel the charger, explain the two allowed cancellation
reasons and require the user to choose or explicitly confirm one; never map a
delay, damage or association reason on their behalf. Finally locate and return
the sneaker after confirmation. Do not claim any refund has been credited or
calculate a new gift-card balance unless a tool explicitly returns that fact.
""",
    "46": """
Help the user correct the malformed order number through conversation and reads.
Locate the air purifier and robotic vacuum, calculate requested totals only from
tool data, and return exactly those items after confirmation. Report a return
request as requested, not as an already completed refund.
""",
    "59": """
Compare the two pending orders without inventing a processing guarantee. If the
user wants cancellation for a delay, explain that the tool accepts only 'no
longer needed' or 'ordered by mistake' and require the user to choose or confirm
one before calling cancel. Handle the address change separately, restate the
full new address, and obtain explicit confirmation before writing it.
""",
    "85": """
Authenticate and inspect the user's orders before choosing an operation. Locate
the fleece jacket and check its order status. Because the target order is
pending, use a single modify-pending-items operation rather than trying an
exchange. Discover the current and requested variant IDs through read tools,
state the price difference and one-time modification consequence, then obtain
explicit confirmation before the write.
""",
    "109": """
Authenticate, inspect the user's orders, and derive the new address only from
the order already sent there. Identify the pending order still using the old
address. Confirm the full address before updating the order and default profile.
For the tablet, enumerate all available variants of the same product, compare
prices, and select the cheapest valid one. Confirm all item and payment details
before the single modify-items write.
""",
}


SYSTEM_PROMPT = """
<instructions>
You are a customer service agent. Follow the policy and the private semantic
plan. In each turn, either send one user message or make one tool call, never
both. Always generate valid JSON for tool calls.

{global_rules}
</instructions>
<policy>
{domain_policy}
</policy>
<private_semantic_plan>
{semantic_plan}
</private_semantic_plan>
""".strip()


class SemanticHintAgent(LLMGTAgent):
    @property
    def system_prompt(self) -> str:
        task_id = str(self.task.id)
        if task_id not in SEMANTIC_PLANS:
            raise ValueError(f"No semantic plan for task {task_id}")
        return SYSTEM_PROMPT.format(
            global_rules=GLOBAL_PROCESS_RULES,
            domain_policy=self.domain_policy,
            semantic_plan=SEMANTIC_PLANS[task_id].strip(),
        )

def create_semantic_hint_agent(tools, domain_policy, **kwargs):
    return SemanticHintAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=kwargs.get("task"),
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        provide_function_args=False,
    )


judge_model = os.environ.get("TAU3_NL_ASSERTIONS_MODEL")
if not judge_model:
    raise RuntimeError("TAU3_NL_ASSERTIONS_MODEL must be set")
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = judge_model

registry.register_agent_factory(
    create_semantic_hint_agent,
    "llm_agent_semantic_hint_v2",
    task_filter=lambda task: str(task.id) in SEMANTIC_PLANS,
)

from tau2.cli import main  # noqa: E402


main()
