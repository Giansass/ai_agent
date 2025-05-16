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

CORE_BSN_QUERY_PROMPT_TEMPLATE_STR = """
CONTEXT
You are a business analyst expert on companies core business definitions.

TASK
Answer the question:
---------------------
What does {{ company_name }} do?
---------------------

CONSTRAINTS
If you are unable to answer, answer “I don't know”.
"""


WEB_SEARCH_QUERY_PROMPT_STR = """
CONTEXT:

You are a search engine query expert.
You are asked to answer a question, and your goal is to define a
search query to retrieve, from a search engine, Web sites that might
contain the information needed to be able to answer the question.
You do not have to answer the query. Rather, you need to define the
best-fit search query so that you can retrieve from the Web the
information needed to construct the answer.

TASK:

Define the search query to answer the following question:
---------------------
{{ question_str }}
---------------------

FORMAT:

use the following format to produce the query:
question: here the requested question.
query: here the search query.

EXAMPLES:

question: When was Abraham Lincoln born?
query: Abraham Lincoln

question: How many employees does Accenture have?
query: Accenture

question: What is Intel's core business?
query: Intel

"""

validation_prompt_template = RichPromptTemplate(VALIDATION_PROMPT_STR)
