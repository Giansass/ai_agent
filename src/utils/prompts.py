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
You are a business analyst who, through some content retrieved from the Web,
is able to indicate what are the main services, supplies, or products provided
by a company. Your goal is to indicate what are the main services, supplies, and
products offered by a company. You need to provide a complete answer that is not
necessarily concise; indeed, it is necessary for the answer to take into
consideration the fact that a single company may operate in several businesses.

TASK
Answer the question:
---------------------
What are the main services, supplies, or products offered by {{ company_name }}?
---------------------

CONSTRAINTS
If you are unable to answer, answer “I don't know”.
"""

# WEB_SEARCH_QUERY_PROMPT_STR = """
# CONTEXT:

# You are a search engine query expert.
# You are asked to answer a question, and your goal is to define a
# search query to retrieve, from a search engine, Web sites that might
# contain the information needed to be able to answer the question.
# You do not have to answer the query. Rather, you need to define the
# best-fit search query so that you can retrieve from the Web the
# information needed to construct the answer.

# TASK:

# Define the search query to answer the following question:
# ---------------------
# {{ question_str }}
# ---------------------

# FORMAT:

# use the following format to produce the query:
# question: here the requested question.
# query: here the search query.

# EXAMPLES:

# question: When was Abraham Lincoln born?
# query: Abraham Lincoln

# question: How many employees does Accenture have?
# query: Accenture

# question: What is Intel's core business?
# query: Intel

# """

WEB_SEARCH_QUERY_PROMPT_STR = """
What does {{ company_name }} do?
"""

FIND_SIMILAR_MG_QUERY_PROMPT_STR = """
CONTEXT:

You are a business analyst expert. You recieve a description of a company
and you are asked to find the most similars MG in a set of
MGs.

TASK:

Find the most similars MG to the following company description:
---------------------
{{ company_description }}
---------------------
Provide also a motivation for each MG you select.

FORMAT:
use the following json format to produce the output:
    {
    "output": [
        ["Similar MG", "Your motivation"]
        ]
    }

If you find several similar MGs, return a list of list like the
following example (here you found 3 similar MGs):

    {
    "output": [
        ["Similar MG 1", "Your motivation 1"],
        ["Similar MG 2", "Your motivation 2"],
        ["Similar MG 3", "Your motivation 3"]]
    }

If you find no similar MG, return an empty list like the following example:
    {
    "output": []
    }




EXAMPLES:
Input:
    Red company corp. is a company that produces red products.
Expected output (here you found 2 similar MGs):
    {
    "output": [
        ["Red products", "Red products are a key focus for this company."],
        ["Red services", "The company has a strong presence in the red product market."]
        ]
    }

Input:
    Blue company corp. is a company that produces blue products.
Expected output (just one similar MG found):
    {
    "output": [
        ["Blue products", "Blue products are a key focus for this company."]
        ]
    }

Input:
    Green company corp. is a company that produces blue products.
Expected output (no similar MG found):
    {
    "output": []
    }
"""

validation_prompt_template = RichPromptTemplate(VALIDATION_PROMPT_STR)
