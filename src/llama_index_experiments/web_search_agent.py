"""aaa"""
import chromadb
from duckduckgo_search import DDGS
from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.vector_stores.chroma import ChromaVectorStore
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from src.llama_index_experiments.llm_load import llm_text_embedding, llm_text_generation
from src.utils.env_var_loader import PHOENIX_COLLECTOR_ENDPOINT
from src.utils.prompts import (
    CORE_BSN_QUERY_PROMPT_TEMPLATE_STR,
    WEB_SEARCH_QUERY_PROMPT_STR,
)

# Set phoenix observability monitor
tracer_provider = register(
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    project_name="Web search agent")

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Create prompts
core_bsn_query_prompt_tmpl = RichPromptTemplate(CORE_BSN_QUERY_PROMPT_TEMPLATE_STR)
web_search_query_prompt_tmpl = RichPromptTemplate(WEB_SEARCH_QUERY_PROMPT_STR)


# Event definitions
class _StartEvent(StartEvent):
    """aaa"""

    company_name: str


class _StopEvent(StopEvent):
    """aaa"""

    query: str


class _SearchEvent(Event):
    """aaa"""

    query: str


class CompanySearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output"""

    company_name: str = Field(description="The company name")
    company_core_business: str = Field(description="The company core business")


pydantic_output_parser = PydanticOutputParser(
    output_cls=CompanySearchQueryDefinitionFormat
)


class WebSearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output"""

    question: str = Field(description="The question")
    query: str = Field(description="The search engine query")


# set up ChromaVectorStore and load in data
db2 = chromadb.PersistentClient(path=".storage/chroma/")
chroma_collection = db2.get_or_create_collection("sample_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=llm_text_embedding)
query_engine = index.as_query_engine(
    similarity_top_k=3,
    vector_store_query_mode="default",
    llm=llm_text_generation,
    output_parser=pydantic_output_parser
)


class WebSearchWorkflow(Workflow):
    """aaa"""

    @step()
    async def get_core_business_from_index(
            self,
            ev: _StartEvent,

    ) -> _SearchEvent:

        """aaa"""

        retr_output = query_engine.query(
            core_bsn_query_prompt_tmpl.format(
                company_name=ev.company_name,
                format_instructions=pydantic_output_parser.get_format_string()
            ))

        print(retr_output)

        return _SearchEvent(query="Exit - correct")

    @step()
    async def get_web_content(
        self, ev: _SearchEvent
    ) -> _SearchEvent:
        """aaa"""

        print(ev.query)
        DDGS(verify=False).text("What does Ferrari do?", max_results=2)

        # query = ev.query
        # web_search_query = llm \
        #     .as_structured_llm(WebSearchQueryDefinitionFormat)\
        #     .complete(web_search_query_prompt_tmpl.format(question_str=query))\
        #     .raw
        #
        # print(f'question: {web_search_query.question}')
        # print(f'query: {web_search_query.query}')

        return _SearchEvent(query="Exit - correct")

    @step()
    async def get_web_results(
            self, ev: _SearchEvent
    ) -> _StopEvent:
        """aaa"""

        query = ev.query
        web_search_query = llm_text_generation \
            .as_structured_llm(WebSearchQueryDefinitionFormat) \
            .complete(web_search_query_prompt_tmpl.format(question_str=query)) \
            .raw

        print(f'question: {web_search_query.question}')
        print(f'query: {web_search_query.query}')

        return _StopEvent(query="Exit - correct")


async def web_search_workflow_execution():
    """Main function"""
    w = WebSearchWorkflow(timeout=10, verbose=False)
    response = await w.run(company_name="Ferrari")
    print(response)
