import os
from typing import TypedDict
import uuid
from langchain.agents import create_agent
import unittest
from unittest import TestCase
from langchain.messages import AIMessage, ToolCall
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
# https://docs.langchain.com/oss/python/langchain/test/unit-testing
from langchain_core.language_models.fake_chat_models import FakeChatModel, GenericFakeChatModel


from early_return_middleware import EarlyReturnMiddleware, EarlyReturnState, skip_llm_call_tool_name


class FakeToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


if not os.environ.get("OPENROUTER_API_KEY"):
    raise Exception("Test expects a valid OPENROUTER_API_KEY for LLM calls")
real_model = ChatOpenRouter(
    model="google/gemini-2.5-flash", api_key=os.environ["OPENROUTER_API_KEY"])


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
        model = FakeToolCallingModel(messages=iter([
            AIMessage(content="", tool_calls=[
                ToolCall(name=skip_llm_call_tool_name, args={
                }, id="1"),
                ToolCall(name=read_mail.name, args={}, id="2"), ]),
            AIMessage(
                content="This message should not be reached with EarlyReturn")
        ]))
        agent = create_agent(model=model, checkpointer=InMemorySaver(), tools=[
            read_mail], middleware=[EarlyReturnMiddleware()])
        res = agent.invoke(
            {"messages": [("user", "Display email content")]}, config)
        last_msg = res["messages"][-1]
        pretty_print_msg(last_msg)
        self.assertTrue(last_msg.content.startswith(
            "Output of tool read_mail:"), last_msg.content)

    def test_long_conversation(self):
        config = get_config()
        model = FakeToolCallingModel(messages=iter([
            AIMessage(content="", tool_calls=[
                # Normally this won't happen because the tool is not even added
                # But here we test if the skip logic is triggered any way
                # TODO: find a way to test the list of tools passed to the LLM and confirm that "skip" is removed
                ToolCall(name=skip_llm_call_tool_name, args={
                }, id="1"),
                ToolCall(name=read_mail.name, args={}, id="2"), ]),
            AIMessage(content="Done")
        ]))
        agent = create_agent(model=model, checkpointer=InMemorySaver(), tools=[
            read_mail], middleware=[EarlyReturnMiddleware(disable_after_messages=2)])
        res = agent.invoke(
            {"messages": [("user", "Hey!"), ("assistant", "Hi"), ("user", "how are you?"), ("assistant", "Fine, what about you?"), ("user", "great, no read my email please")]}, config)
        last_msg = res["messages"][-1]
        pretty_print_msg(last_msg)
        self.assertTrue(last_msg.content.startswith(
            "Done"), last_msg.content)

    @unittest.skip("Costs money")
    def test_single_call(self):
        config = get_config()
        agent = create_agent(model=real_model, checkpointer=InMemorySaver(), tools=[
            read_mail], middleware=[EarlyReturnMiddleware()])
        res = agent.invoke(
            {"messages": ["Display the content of the mail"]}, config)
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
        self.assertEqual(last_msg.content,
                         "Output of tool read_mail:\n Hello!")

    @unittest.skip("Costs money")
    def test_longer_operation(self):
        """
        As soon as there are more than one tool, early_return is disabled

        If you want the display tool to always skip the last LLM call,
        early return middleware is not the right solution,
        instead you should use the "return_direct" option on the tool
        """
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        agent = create_agent(model=real_model, checkpointer=InMemorySaver(), tools=[read_mail, reply], middleware=[
            EarlyReturnMiddleware()])
        res = agent.invoke(
            {"messages": ["Read mail, then reply to it automatically with 'Hello you !'."]}, config)
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
