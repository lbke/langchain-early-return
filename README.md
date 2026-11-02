## Early Return Middleware

This middleware is an implementation of the "Dynamic Direct Return" pattern. 

The idea is to shortcut the final LLM call of the agentic loop, which is often useless for short-running loop in agentic contexts.

Example:
- Opening a file
- Reading an email

Implementation is based on a special tool used as a flag raised by the LLM.

The added value of this pattern decrease as the conversation grows, therefore this implementation will disable the middleware after a few AI messages in a thread. Hence the `EarlyReturn` name for this implementation of DDR. 

A variation of this early return pattern could choose to disable the middleware based on turns in the current loop invoke call, rather than based on the whole conversation length as it is currently implemented. This means that the dynamic direct return could be triggered multiple time in a single conversation.

Edge cases and alternatives:
- If a tool should always bypass the last LLM call, use the [return_direct](https://reference.langchain.com/python/langchain-core/tools/base/BaseTool/return_direct) option directly on the tool instead of this middleware
- If a tool returns an error, you'd want the LLM to process it. We detect errors via [`ToolMessage.status == error`](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage/status) and expects error messages to be properly marked by the tool.
- Tools that return commands currently do not seem to work properly when the middleware kicks in.

Article (in French) about the rationale behind DDR: https://www.lbke.fr/formations/ressources/langchain/dynamic-direct-return-optimiser-la-boucle-agentique
Feature request for Goose : https://github.com/aaif-goose/goose/issues/12639
Previous implementation for LBKE's `fais` agent : https://github.com/aaif-goose/goose/issues/12639

## Test

uv run python -m unittest -v

## Roadmap

- Exchange with LangChain contributors on this idea
- Run against real agentic benchmarks to assess the impact
- Move tests to pytest, split unit and integration
- Add a `disable_after_turns` option to implement deactivation based on the current number of turns for the agentic loop (rather than conversation length with `disable_after_messages`)
- Fix the edge case of Command-returning tools