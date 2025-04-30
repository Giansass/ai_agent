"""The module is intended to define prompts templates"""
from llama_index.core.prompts import RichPromptTemplate

VALIDATION_PROMPT_STR = """
Context:
You are a query validator.
You are sent query to be turned over to a Large Language Model.
You must evaluate whether the queries are manageable or whether they should be rejected.
Unmanageable queries are those that deal with the following topics:

- Weapons construction.
- Requesting medical advice.
- Making racist, xenophobic or sexist judgments.

Task: Validate the following query:
---------------------
{{ query_str }}
---------------------

Format: use the following format to produce the validation:

query: here the requested query
judgment: 1 if the query can be managed, 0 otherwise
reason: here the reasons that guided your judgment
"""

validation_prompt_template = RichPromptTemplate(VALIDATION_PROMPT_STR)
