"""aaa"""
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from src.llama_index_experiments.llm_load import llm
from src.utils.env_var_loader import PHOENIX_COLLECTOR_ENDPOINT
from src.utils.prompts import WEB_SEARCH_QUERY_PROMPT_STR

# Set phoenix observability monitor
tracer_provider = register(
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    project_name="Web search agent")

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


# Create prompts
web_search_query_prompt_tmpl = RichPromptTemplate(WEB_SEARCH_QUERY_PROMPT_STR)


# Event definitions
class _StartEvent(StartEvent):
    """aaa"""

    query: str


class _StopEvent(StopEvent):
    """aaa"""

    query: str


class _SearchEvent(Event):
    """aaa"""

    query: str


class WebSearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output"""

    question: str = Field(description="The question")
    query: str = Field(description="The search engine query")


class WebSearchWorkflow(Workflow):
    """aaa"""

    @step()
    async def get_query(
        self, ev: _StartEvent
    ) -> _SearchEvent:
        """aaa"""

        query = ev.query
        web_search_query = llm \
            .as_structured_llm(WebSearchQueryDefinitionFormat)\
            .complete(web_search_query_prompt_tmpl.format(question_str=query))\
            .raw

        print(f'question: {web_search_query.question}')
        print(f'query: {web_search_query.query}')

        return _SearchEvent(query="Exit - correct")

    @step()
    async def get_web_results(
            self, ev: _SearchEvent
    ) -> _StopEvent:
        """aaa"""

        query = ev.query
        web_search_query = llm \
            .as_structured_llm(WebSearchQueryDefinitionFormat) \
            .complete(web_search_query_prompt_tmpl.format(question_str=query)) \
            .raw

        print(f'question: {web_search_query.question}')
        print(f'query: {web_search_query.query}')

        return _StopEvent(query="Exit - correct")


async def web_search_workflow_execution():
    """Main function"""
    w = WebSearchWorkflow(timeout=10, verbose=False)
    response = await w.run(query="What is Ferrari's core business?")
    print(response)
