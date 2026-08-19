"""VERL 0.8 multi-turn AgentLoop for the frozen tau3 retail pilot.

This module is intentionally a thin specialization of VERL's native
``ToolAgentLoop``.  It does not implement GRPO loss, FSDP, inference, or weight
loading.  The latter remain VERL/vLLM responsibilities.
"""

from __future__ import annotations

import json
from uuid import uuid4

from .config import FROZEN_CONFIG
from .rollout_bridge import Tau3RolloutSession, render_tools_for_prompt

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
        messages = [
            {
                "role": "system",
                "content": reset_result.system_prompt + render_tools_for_prompt(reset_result.tools),
            },
            {"role": "user", "content": reset_result.observation},
        ]
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
                extra_fields={"official_terminal_reward": session.terminal_reward.value if session.terminal_reward else 0.0},
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
        state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        if agent_data.tool_calls:
            call = agent_data.tool_calls[0]
            arguments = call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            assistant_message = json.dumps(
                {"name": call.name, "arguments": arguments}, ensure_ascii=False
            )
        else:
            assistant_message = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
            )
        agent_data.messages.append({"role": "assistant", "content": assistant_message})
        agent_data.tool_calls = []
        return _INTERACTING

    async def _handle_interacting_state(self, agent_data, session):
        assistant_message = agent_data.messages[-1]["content"]
        result = session.submit(assistant_message)
        agent_data.turn_scores.append(result.reward or 0.0)
        if result.observation:
            agent_data.messages.append({"role": "user", "content": result.observation})
            observation_ids = await self.apply_chat_template(
                [{"role": "user", "content": result.observation}],
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += observation_ids
            # These are DeepSeek/user/environment observations, never policy loss.
            agent_data.response_mask += [0] * len(observation_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(observation_ids)
        if result.terminated or result.truncated:
            return AgentState.TERMINATED
        return AgentState.GENERATING
