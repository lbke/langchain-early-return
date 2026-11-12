## Early Return Middleware

This middleware is an implementation of the "Dynamic Direct Return" pattern. 

The idea is to shortcut the final LLM call of the agentic loop, which is often useless for short-running loop in agentic contexts.

Example:
- Opening a file
- Reading an email
- An automation that doesn't deserve afterwards analysis from the LLM.

Implementation is based on a special tool, used as a flag raised by the LLM.

The added value of this pattern decrease as the conversation grows, therefore this implementation will disable the middleware after a few AI messages in a thread. Hence the `EarlyReturn` name for this implementation of DDR. 

Edge cases and alternatives:
- If a tool should always bypass the last LLM call, use the [return_direct](https://reference.langchain.com/python/langchain-core/tools/base/BaseTool/return_direct) option directly on the tool instead of this middleware
- If a tool returns an error, you'd want the LLM to process it. We detect errors via [`ToolMessage.status == error`](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage/status) and expects error messages to be properly marked by the tool.

Article (in French) about the rationale behind DDR: https://www.lbke.fr/formations/ressources/langchain/dynamic-direct-return-optimiser-la-boucle-agentique  
Feature request for Goose : https://github.com/aaif-goose/goose/issues/12639  
Previous implementation for LBKE's `fais` agent : https://github.com/aaif-goose/goose/issues/12639

## Test

### Unit test

```sh
uv run python -m unittest -v
```

### Deep agent eval suite


Install:

```sh
git clone https://github.com/langchain-ai/deepagents
cd deepagents/libs/evals
uv sync
# Set your LANGSMITH_API_KEY etc.
# Deep Agents are supposed to load .env automatically
# https://docs.langchain.com/oss/deepagents/code/configuration#environment-variables
```

```sh
# .env
LANGSMITH_API_KEY=XXX
LANGSMITH_PROJECT=early-return-midleware
LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
OPENROUTER_API_KEY=XXX
```

Run without EarlyReturnMiddleware:

```sh
LANGSMITH_PROJECT=eval_early_return LANGSMITH_TRACING_V2=true LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals/test_file_operations.py --model openrouter:deepseek/deepseek-v4-flash
# In LangSmith, set as baseline
```

Run with EarlyReturnMiddleware:

```sh
cp early_return_middleware.py deepagents/libs/evals/test
# MANUAL SETUP: update test_file_operation to use the middleware with "never_disable=True"
LANGSMITH_PROJECT=eval_early_return LANGSMITH_TRACING_V2=true LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals/test_file_operations.py --model openrouter:deepseek/deepseek-v4-flash
# In Langsmith : tag the new eval with "early_return=true"
```

## Roadmap

- Exchange with LangChain contributors on this idea
- Run against real agentic benchmarks to assess the impact
- Move tests to pytest, split unit and integration
- Add a `disable_after_turns` option to implement deactivation based on the current number of turns for the agentic loop (rather than conversation length with `disable_after_messages`)
- Fix the edge case of Command-returning tools
- Tools that return LangGraph `Command` currently do not seem to work properly when the middleware kicks in.
- Make it easier to reuse Deep Agents eval suite with a different middleware harness

A variation of this early return pattern could choose to disable the middleware based on turns in the current loop invoke call, rather than based on the whole conversation length as it is currently implemented. This means that the dynamic direct return could be triggered multiple time in a single conversation.