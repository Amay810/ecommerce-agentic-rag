"""VERL 0.8 multi-turn AgentLoop for the frozen tau3 retail pilot.

This module is intentionally a thin specialization of VERL's native
``ToolAgentLoop``.  It does not implement GRPO loss, FSDP, inference, or weight
loading.  The latter remain VERL/vLLM responsibilities.
"""

from __future__ import annotations

from uuid import uuid4

from .config import FROZEN_CONFIG
from .rollout_bridge import Tau3RolloutSession
from .tool_channel import (
    assistant_message_from_function_call,
    gym_action_from_function_call,
    initial_agent_messages,
    messages_from_gym_observation,
)

try:  # Keep offline VM checks importable without the NSCC training stack.
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
    from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
    _VERL_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only on VM/CI
    AgentLoopOutput = object  # type: ignore[assignment,misc]
    AgentData = object  # type: ignore[assignment,misc]
    ToolAgentLoop = object  # type: ignore[assignment,misc]
    AgentState = object  # type: ignore[assignment,misc]
    _VERL_IMPORT_ERROR = exc

    def register(_name):
        return lambda cls: cls


_INTERACTING = object()


def _bounded_sampling_params(sampling_params: dict, *, used_tokens: int, response_length: int) -> dict:
    available = response_length - used_tokens
    if available <= 0:
        raise RuntimeError(
            "VERL response budget exhausted before tau2 reached a terminal state"
        )
    bounded = dict(sampling_params)
    configured = bounded.get("max_tokens")
    if configured is None or int(configured) > available:
        bounded["max_tokens"] = available
    return bounded


@register("tau3_agent")
class Tau3AgentLoop(ToolAgentLoop):
    """Drive fresh tau2 episodes while retaining VERL token accounting."""

    def __init__(self, *args, **kwargs):
        if _VERL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "VERL 0.8 is required on the NSCC training runtime"
            ) from _VERL_IMPORT_ERROR
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict, **kwargs):
        extra_info = kwargs.get("extra_info") or {}
        tau_root = extra_info["tau_root"]
        task_id = str(extra_info["task_id"])
        user_base_url = extra_info["user_base_url"]
        session = Tau3RolloutSession(
            tau_root=tau_root, task_id=task_id, user_base_url=user_base_url
        )
        reset_result = session.start()
        tool_schemas = list(reset_result.tools)
        if not tool_schemas:
            raise RuntimeError(
                "tau2 reset returned no tool schemas; Qwen chat template "
                "cannot take the tools= branch"
            )
        messages = initial_agent_messages(
            reset_result.system_prompt, reset_result.observation
        )
        request_id = uuid4().hex
        agent_data = AgentData(
            messages=messages,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs={},
            metrics={},
            request_id=request_id,
            tools_kwargs={},
        )
        # Parent pending applies apply_chat_template(..., tools=schemas).
        # These are schema dicts for the Qwen template only — not VERL BaseTool
        # executors. Gym remains the only tool runtime.
        agent_data._active_tool_schemas = tool_schemas
        state = AgentState.PENDING
        try:
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state is _INTERACTING:
                    state = await self._handle_interacting_state(agent_data, session)
                else:
                    raise RuntimeError(f"unexpected VERL AgentState: {state!r}")

            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=agent_data.response_mask[: self.response_length],
                multi_modal_data={},
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
                response_logprobs=(
                    agent_data.response_logprobs[: self.response_length]
                    if agent_data.response_logprobs else None
                ),
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={
                    "official_terminal_reward": (
                        session.terminal_reward.value
                        if session.terminal_reward
                        else 0.0
                    ),
                    "reward_extra_info": {
                        "source": "tau2.evaluator.evaluate_simulation",
                        "evaluation_type": FROZEN_CONFIG.evaluation_type,
                    },
                },
            )
            if session.terminal_reward is None:
                raise RuntimeError("VERL rollout completed without an official terminal reward")
            # VERL consumes reward_score for GRPO advantages. The value is the
            # official terminal reward only; no dense/tool/action proxy enters.
            output.reward_score = float(session.terminal_reward.value)
            return output
        finally:
            session.close()

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        sampling_params = _bounded_sampling_params(
            sampling_params,
            used_tokens=len(agent_data.response_mask),
            response_length=self.response_length,
        )
        await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        if agent_data.tool_calls:
            call = agent_data.tool_calls[0]
            gym_action = gym_action_from_function_call(call.name, call.arguments)
            assistant_message = assistant_message_from_function_call(
                call.name, call.arguments
            )
        else:
            gym_action = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
            )
            assistant_message = {"role": "assistant", "content": gym_action}
        agent_data._pending_gym_action = gym_action
        agent_data.messages.append(assistant_message)
        agent_data.tool_calls = []
        return _INTERACTING

    async def _handle_interacting_state(self, agent_data, session):
        gym_action = getattr(agent_data, "_pending_gym_action", None)
        if not gym_action:
            raise RuntimeError("Tau3AgentLoop interacting without a gym action")
        result = session.submit(gym_action)
        agent_data.turn_scores.append(result.reward or 0.0)
        new_messages = messages_from_gym_observation(result.observation)
        if new_messages:
            agent_data.messages.extend(new_messages)
            # Do not pass tools= here: the Qwen template would re-inject the
            # # Tools system block. Parent ToolAgentLoop tokenizes tool
            # responses the same way. Sampled assistant tokens stay as-is;
            # retokenizing them would replace GRPO tokens with template tokens.
            observation_ids = await self.apply_chat_template(
                new_messages,
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += observation_ids
            # These are DeepSeek/user/environment observations, never policy loss.
            agent_data.response_mask += [0] * len(observation_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(observation_ids)
            agent_data.user_turns += 1
        if result.terminated or result.truncated:
            return AgentState.TERMINATED
        return AgentState.GENERATING
