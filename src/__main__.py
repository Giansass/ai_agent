"""Entry point for the src."""

from src.utils.env_var_loader import AGENT_TO_BE_EXECUTED

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    if AGENT_TO_BE_EXECUTED == "Simple agent":
        from src.llama_index_experiments.simple_agent import simple_agent_execution

        asyncio.run(simple_agent_execution())

    elif AGENT_TO_BE_EXECUTED == "Yahoo Agent":
        from src.llama_index_experiments.yahoo_search_agent import yahoo_agent_execution

        asyncio.run(yahoo_agent_execution())

    elif AGENT_TO_BE_EXECUTED == "Simple workflow":
        from src.llama_index_experiments.simple_workflow import workflow_execution

        asyncio.run(workflow_execution())

    # elif AGENT_TO_BE_EXECUTED == "HF workflow":
    #     from src.llama_index_experiments.hf_workflow import workflow_execution
    #
    #     asyncio.run(workflow_execution())

    elif AGENT_TO_BE_EXECUTED == "HF simple workflow":
        from src.llama_index_experiments.hf_simple_workflow import main

        asyncio.run(main())

    else:
        raise ValueError(f"Agent {AGENT_TO_BE_EXECUTED} not supported")
