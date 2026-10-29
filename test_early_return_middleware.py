import os
from typing import TypedDict
import uuid
from langchain.agents import create_agent
from unittest import TestCase
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

from early_return_middleware import EarlyReturnMiddleware

if not os.environ.get("OPENROUTER_API_KEY"):
    raise Exception("Test expects a valid OPENROUTER_API_KEY for LLM calls")

model = ChatOpenRouter(model="google/gemini-2.5-flash",
                       api_key=os.environ["OPENROUTER_API_KEY"])


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


class TestEarlyReturn(TestCase):
    def test_single_call(self):
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        agent = create_agent(model=model, checkpointer=InMemorySaver(), tools=[
                             read_mail], middleware=[EarlyReturnMiddleware()])
        res = agent.invoke(
            {"messages": ["Display the content of the mail"]}, config)
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
        self.assertEqual(last_msg.content,
                         "Output of tool read_mail:\n Hello!")

    def test_longer_operation(self):
        """
        As soon as there are more than one tool, early_return is disabled

        If you want the display tool to always skip the last LLM call,
        early return middleware is not the right solution,
        instead you should use the "return_direct" option on the tool
        """
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        agent = create_agent(model=model, checkpointer=InMemorySaver(), tools=[read_mail, reply], middleware=[
                             EarlyReturnMiddleware()])
        res = agent.invoke(
            {"messages": ["Read mail, then reply to it automatically with 'Hello you !'."]}, config)
        last_msg = res["messages"][-1]
        print("\n---")
        print(last_msg.pretty_print())
        print("\n---")
