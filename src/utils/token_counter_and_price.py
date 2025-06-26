"""a module to print token usage and cost statistics."""

from src.utils.env_var_loader import (
    COMPLETITION_TOKEN_DOLLAR_PRICE_PER_1M,
    EMBEDDING_TOKEN_DOLLAR_PRICE_PER_1M,
    PROMPT_TOKEN_DOLLAR_PRICE_PER_1M,
)


def print_token_usage(token_counter):
    """Prints the token usage statistics from the provided token counter."""

    _1m = 1_000_000

    emb_tkn = token_counter.total_embedding_token_count
    prompt_tkn = token_counter.prompt_llm_token_count
    completion_tkn = token_counter.completion_llm_token_count

    e_price = emb_tkn*EMBEDDING_TOKEN_DOLLAR_PRICE_PER_1M/_1m
    print(f"Embedding Tokens: {emb_tkn}")
    print(f"Embedding Token Price: {e_price:.4}$")

    i_price = prompt_tkn*PROMPT_TOKEN_DOLLAR_PRICE_PER_1M/_1m
    print(f"Input Tokens: {prompt_tkn}")
    print(f"Input Token Price: {i_price:.4}$")

    o_price = completion_tkn*COMPLETITION_TOKEN_DOLLAR_PRICE_PER_1M/_1m
    print(f"Output Tokens: {completion_tkn}")
    print(f"Output Token Price: {o_price:.4}$")

    total_price = e_price + i_price + o_price
    print(f"Total Tokens: {token_counter.total_llm_token_count}")
    print(f"Total Price: {total_price:.4}$")
