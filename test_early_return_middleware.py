import os
from typing import TypedDict
import uuid
from langchain.agents import AgentState, create_agent
import unittest
from unittest import TestCase
from langchain.messages import AIMessage, ToolCall, ToolMessage
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

# https://docs.langchain.com/oss/python/langchain/test/unit-testing
from langchain_core.language_models.fake_chat_models import (
    FakeChatModel,
    GenericFakeChatModel,
)
from langgraph.runtime import Runtime
from langgraph.types import Command


from early_return_middleware import (
    EarlyReturnMiddleware,
    EarlyReturnState,
    skip_llm_call_tool_name,
)


class FakeToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


if not os.environ.get("OPENROUTER_API_KEY"):
    raise Exception("Test expects a valid OPENROUTER_API_KEY for LLM calls")
real_model = ChatOpenRouter(
    model="google/gemini-2.5-flash", api_key=os.environ["OPENROUTER_API_KEY"]
)


@tool
def read_mail() -> str:
    """Tool to read a mail"""
    return "Hello!"


@tool
def reply(reply: str) -> str:
    """Tool to replay to a mail"""
    return "Reply sent with success"


class Context(TypedDict):
    pass


def pretty_print_msg(msg):
    print("\n---")
    print(msg.pretty_print())
    print("\n---")


def get_config():
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    return config


class TestEarlyReturn(TestCase):
    def test_early_return(self):
        config = get_config()
        model = FakeToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            ToolCall(name=skip_llm_call_tool_name,
                                     args={}, id="1"),
                            ToolCall(name=read_mail.name, args={}, id="2"),
                        ],
                    ),
                    AIMessage(
                        content="This message should not be reached with EarlyReturn"
                    ),
                ]
            )
        )
        agent = create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            tools=[read_mail],
            middleware=[EarlyReturnMiddleware()],
        )
        res = agent.invoke(
            {"messages": [("user", "Display email content")]}, config)
        last_msg = res["messages"][-1]
        pretty_print_msg(last_msg)
        self.assertTrue(
            last_msg.content.startswith(
                "Output of tool read_mail:"), last_msg.content
        )

    def test_long_conversation(self):
        """
        Do not trigger dynamic direct return if conversation goes longer than configured lenght
        """
        config = get_config()
        model = FakeToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            # Normally this won't happen because the tool is not even added
                            # But here we test if the skip logic is triggered any way
                            # TODO: find a way to test the list of tools passed to the LLM and confirm that "skip" is removed
                            ToolCall(name=skip_llm_call_tool_name,
                                     args={}, id="1"),
                            ToolCall(name=read_mail.name, args={}, id="2"),
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )
        agent = create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            tools=[read_mail],
            middleware=[EarlyReturnMiddleware(disable_after_messages=2)],
        )
        res = agent.invoke(
            {
                "messages": [
                    ("user", "Hey!"),
                    ("assistant", "Hi"),
                    ("user", "how are you?"),
                    ("assistant", "Fine, what about you?"),
                    ("user", "great, no read my email please"),
                ]
            },
            config,
        )
        last_msg = res["messages"][-1]
        pretty_print_msg(last_msg)
        self.assertTrue(last_msg.content.startswith("Done"), last_msg.content)

    def test_errored_tool(self):
        """
        Do not trigger if at least one tool is errored
        """
        config = get_config()

        @tool
        def read_mail_error(runtime: Runtime):
            """
            Read an email - may trigger errors if mail doesn't exist anymore
            """
            raise ValueError("Mail not found")
        model = FakeToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            # Normally this won't happen because the tool is not even added
                            # But here we test if the skip logic is triggered any way
                            # TODO: find a way to test the list of tools passed to the LLM and confirm that "skip" is removed
                            ToolCall(name=skip_llm_call_tool_name,
                                     args={}, id="1"),
                            ToolCall(name=read_mail_error.name,
                                     args={}, id="2"),
                        ],
                    ),
                    AIMessage(
                        content="Error with read_mail tool, try again letter."),
                ]
            )
        )
        agent = create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            tools=[read_mail_error],
            middleware=[EarlyReturnMiddleware()],
        )
        res = agent.invoke(
            {
                "messages": [
                    ("user", "Read the email"),
                ]
            },
            config,
        )
        last_msg = res["messages"][-1]
        pretty_print_msg(last_msg)
        self.assertTrue(
            last_msg.content.startswith(
                "Error with read_mail tool"), last_msg.content
        )

    @unittest.skip("Not implemented yet")
    def test_command_output(self):
        """
        LangChain tool-level return_direct has to take the final state update into account,
        so should we with dynamic direct return
        https://docs.langchain.com/oss/python/langchain/tools#return-a-command-with-return_direct

        NOTE: this tests doesn't pass yet, the middleware needs to somehow handle this edge case
        """
        config = get_config()

        @tool
        def read_mail_cmd(runtime: Runtime):
            """
            Read an email and update state to confirm read
            """
            return Command(
                update={
                    "successful_read": True,
                    "messages": [
                        ToolMessage(content="Hello!",
                                    tool_call_id=runtime.tool_call_id)
                    ],
                }
            )
        model = FakeToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            ToolCall(name=skip_llm_call_tool_name,
                                     args={}, id="1"),
                            ToolCall(name=read_mail_cmd.name,
                                     args={}, id="2"),
                        ],
                    ),
                    AIMessage(
                        content="This message should not be reached with EarlyReturn"
                    ),
                ]
            )
        )

        class ReadSuccessState(AgentState):
            successful_read: bool = False
        agent = create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            tools=[read_mail_cmd],
            middleware=[EarlyReturnMiddleware()],
            state_schema=ReadSuccessState,
        )
        res = agent.invoke(
            {"messages": ["Display the content of the mail"], "successful_read": False}, config)
        print(res)
        self.assertTrue(res["successful_read"],
                        "State should be updated to successful_read=True")

    @unittest.skip("Costs money")
    def test_single_call(self):
        config = get_config()
        agent = create_agent(
            model=real_model,
            checkpointer=InMemorySaver(),
            tools=[read_mail],
            middleware=[EarlyReturnMiddleware()],
        )
        res = agent.invoke(
            {"messages": ["Display the content of the mail"]}, config)
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
        self.assertTrue(
            last_msg.content.startswith(
                "Output of tool read_mail:"), last_msg.content
        )

    @unittest.skip("Costs money")
    def test_longer_operation(self):
        """
        As soon as there are more than one tool, early_return is disabled

        If you want the display tool to always skip the last LLM call,
        early return middleware is not the right solution,
        instead you should use the "return_direct" option on the tool
        """
        config = get_config()
        agent = create_agent(
            model=real_model,
            checkpointer=InMemorySaver(),
            tools=[read_mail, reply],
            middleware=[EarlyReturnMiddleware()],
        )
        res = agent.invoke(
            {
                "messages": [
                    "Read mail, then reply to it automatically with 'Hello you !'."
                ]
            },
            config,
        )
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
        self.assertFalse(
            last_msg.content.startswith("Output of tool"), last_msg.content
        )
