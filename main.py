from typing import Callable
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware, AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

skip_llm_call_tool_name="skip_llm_call"

@tool(name=skip_llm_call_tool_name)
def skip_llm_call():
    """
    Use this tool in parallel to other tools whenever you think the tools direct output will be enough to answer the user questions
    This reduces costs by avoiding a useless LLM inference
    - If multiple tools must be called in a sequence call only for the LAST tool in the sequence
    - Use this tool in parallel with the tools that you want to return directly without being read by the LLM, not AFTER (otherwise it is useless)
    - Never call this tool alone, it should always be called with other tools, otherwise it doesn't make sense
    """
    return


class EarlyReturn(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        msgs=request.state["messages"]
        # System + User + AI or User + AI + User => not in an "early" return case
        if len(msgs) >=3
            return handler(request)
        # User + AI => do not add tool
        if len(msgs) == 2 and msgs[-1].type == "ai":
            return handler(request)
        # System + AI or User => must add tool
        request.tools.append(skip_llm_call)
        return handler(request)

    def after_model(self, state: AgentState, runtime: Runtime):
        """
        Detect LLM calls to the return_direct tool and set a flag in the context
        Alternative : parse messages everytime in apply_return_direct,
        but it would be more compute intensive
        """
        last_msg = state["messages"][-1]
        # Detect a call to "return_direct" tool
        if isinstance(last_msg, AIMessage):
            return_direct = next(
                (tc for tc in last_msg.tool_calls if tc["name"] == skip_llm_call_tool_name),
                None,
            )
            if return_direct:
                # unknow state field are dismissed so use context instead to flag return_direct calls
                runtime.context["__early_return_called"] = True
                # must track how many tools should be called before we stop the loop (to handle parallel calls)
                runtime.context["__early_return_expected_tool_calls"] = len(
                    last_msg.tool_calls
                )
        return None

    def before_model(self, state: AgentState, runtime: Runtime):
        """
        Skip the final LLM call if early return has been triggered
        """
        # If return_direct was called
        if not runtime.context.get("__early_return_called"):
            return None
        expected_tool_calls = runtime.context.get("__early_return_expected_tool_calls")
        # Get latest tool calls
        idx = 1
        tool_msgs = []
        while idx <= len(state["messages"]):
            msg = state["messages"][-idx]
            if isinstance(msg, ToolMessage):
                tool_msgs.append(msg)
                idx += 1
            else:
                break
        # We are still waiting for some tool messages
        if len(tool_msgs) < expected_tool_calls:
            return None
        # We have collected all tool calls, we can return the aggregated content
        agg_content = ""
        for msg in tool_msgs:
            # Ignore the return_direct tool itself
            if msg.name == "return_direct":
                continue
            agg_content += f"Output of tool {msg.name}:\n{msg.content}"
        # Alternative implementation would be using wrap_model_call and return "request" with no handler call
        # https://docs.langchain.com/oss/python/langchain/middleware/custom#agent-jumps
        return {"messages": [AIMessage(content=agg_content)], "jump_to": "end"}
