"""Entry point for the src."""

from src.llama_index_experiments.simple_agent import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(main())
