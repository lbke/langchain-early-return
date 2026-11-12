from typing import Callable, Optional
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, ai
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
    # EarlyReturn is disabled as soon as the conversion grows longer
    # Even if the messages list is shrunk later on (summarization for instance)
    early_return_disabled: bool


class EarlyReturnMiddleware(AgentMiddleware):
    state_schema = EarlyReturnState
    tools = [skip_llm_call]
    """
    Enforce the skip LLM call
    NOTE: a middleware can only filter tools dynamically, not add more,
    all tools must be declared upfront hence this variable
    """

    disable_after_messages: int
    """
    For long conversations with complex tasks EarlyReturn brings little token reduction
    So we disable it after N AI messages
    Recommanded value: from 1 to 5
    """

    remove_tool: bool
    """
    Removing a tool from the tools list may break caching
    As a default, the tool is made inactive, but kept around in the cache
    """

    never_disable: bool
    """
    Keep early return active at all time even for long conversations
    """

    def __init__(self, disable_after_messages=1, never_disable=False, remove_tool=False):
        self.disable_after_messages = disable_after_messages
        self.never_disable = never_disable
        self.remove_tool = remove_tool

    def _should_disable_early_return(self, state: EarlyReturnState):
        if self.never_disable:
            return False
        if state.get("early_return_disabled"):
            return True
        ai_msgs = [m for m in state["messages"]
                   if isinstance(m, AIMessage)]
        return len(
            ai_msgs) > self.disable_after_messages

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        """
        Filter out the skip_llm tool under certain conditions

        NOTE: filtering a tool may break cache, so it's not the default behaviour,
        we favour keeping the tool around but disabling it
        """
        if self.remove_tool and self._should_disable_early_return(request.state):
            # Filter out the skip tool to disable EarlyReturn
            tools = [t for t in request.tools if t.name !=
                     skip_llm_call_tool_name]
            return handler(request.override(tools=tools))
        # EarlyReturn is enabled
        return handler(request)

    def after_model(self, state: EarlyReturnState, runtime: Runtime):
        """
        Detect LLM calls to the skip tool and set a flag in the state
        Alternative : parse messages everytime in apply_return_direct,
        but it would be more compute intensive
        """
        if self._should_disable_early_return(state):
            return None
        last_msg = state["messages"][-1]
        # Detect a call to the skip tool
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
        if self._should_disable_early_return(state):
            # Cache early return computation
            # to avoid computing the length of the message list on each turn
            return {
                "early_return_disabled": True
            }
        # If skip tool was not called, skip
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
        # If a lst one tool has an error: never skip the LLM call
        # Errors are detected by "ToolNode" and caught automatically with "status: error" added on the tool message
        # https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage/status
        if any([m for m in tool_msgs if m.status == "error"]):
            return None
        # We are still waiting for some tool messages, let the loop run
        if len(tool_msgs) < expected_tool_calls:
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
