from typing import Callable
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
import logging
logger = logging.getLogger(__name__)


skip_llm_call_tool_name = "skip_llm_call"


@tool(skip_llm_call_tool_name)
def skip_llm_call():
    """
    Use this tool in parallel to other tools whenever you think the tools direct output will be enough to answer the user questions
    This reduces costs by avoiding a useless LLM inference
    - If multiple tools must be called in a sequence call only for the LAST tool in the sequence
    - Use this tool in parallel with the tools that you want to return directly without being read by the LLM, not AFTER (otherwise it is useless)
    - Never call this tool alone, it should always be called with other tools, otherwise it doesn't make sense
    """
    return


class EarlyReturnState(AgentState):
    early_return_called: bool
    early_return_expected_tool_calls: int


class EarlyReturnMiddleware(AgentMiddleware):
    state_schema = EarlyReturnState
    # We can't add tools dynamically, only filter them out
    tools = [skip_llm_call]

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        msgs = request.state["messages"]
        # System + User + AI or User + AI + User => not in an "early" return case
        if len(msgs) >= 3:
            return handler(request)
        # User + AI => do not add tool
        if len(msgs) == 2 and msgs[-1].type == "ai":
            return handler(request)
        # System + AI or User => first message of the conversation, must add the skip_llm tool
        tools = [t for t in request.tools if t.name !=
                 skip_llm_call_tool_name]
        return handler(request.override(tools=tools))

    def after_model(self, state: EarlyReturnState, runtime: Runtime):
        """
        Detect LLM calls to the return_direct tool and set a flag in the state
        Alternative : parse messages everytime in apply_return_direct,
        but it would be more compute intensive
        """
        last_msg = state["messages"][-1]
        # Detect a call to "return_direct" tool
        if isinstance(last_msg, AIMessage):
            early_return = next(
                (tc for tc in last_msg.tool_calls if tc["name"]
                 == skip_llm_call_tool_name),
                None,
            )
            logger.debug(f"Has early return: {bool(early_return)}")
            if early_return:
                return {
                    "early_return_called": True,
                    "early_return_expected_tool_calls": len(
                        last_msg.tool_calls)
                }
        # Always reset, the state is persisted by the checkpointer across invocations
        return None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: EarlyReturnState, runtime: Runtime):
        """
        Skip the final LLM call if early return has been triggered
        """
        # If skip tool was called and detected in after_model
        if not state.get("early_return_called"):
            return None
        expected_tool_calls = state.get(
            "early_return_expected_tool_calls", 0)
        # Get latest tool calls
        idx = 1
        tool_msgs = []
        # Get all tool message to build the final AIMessage manually
        while idx <= len(state["messages"]):
            msg = state["messages"][-idx]
            if isinstance(msg, ToolMessage):
                tool_msgs.append(msg)
                idx += 1
            else:
                break
        # We are still waiting for some tool messages, let the loop run
        if len(tool_msgs) < expected_tool_calls:
            print("Waiting for tool call")
            return None
        # We have collected all tool calls, we can return the aggregated content
        agg_content = ""
        for msg in tool_msgs:
            # Ignore the return_direct tool itself
            if msg.name == skip_llm_call_tool_name:
                continue
            agg_content += f"Output of tool {msg.name}:\n{msg.content}"
        # Alternative implementation would be using wrap_model_call and return "request" with no handler call
        # https://docs.langchain.com/oss/python/langchain/middleware/custom#agent-jumps
        return {
            "messages": [AIMessage(content=agg_content)],
            "early_return_called": None,
            "early_return_expected_tool_calls": None,
            "jump_to": "end",
        }
