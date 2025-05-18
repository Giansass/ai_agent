"""aaa"""
import copy
import pickle
import re

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.readers.web import BeautifulSoupWebReader
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

# Web content loader
web_content_loader = BeautifulSoupWebReader()


# Event definitions
class _StartEvent(StartEvent):
    """aaa"""

    company_name: str


class _StopEvent(StopEvent):
    """aaa"""

    query: str


class _UrlSearchEvent(Event):
    """aaa"""

    query: str


class _UrlContentExtractionEvent(Event):
    """aaa"""

    query: str


class _UrlContentStoringEvent(Event):
    """aaa"""

    query: str


class _GetWebEvent(Event):
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

    ) -> _UrlSearchEvent | _StopEvent:

        """aaa"""

        # retr_output = query_engine.query(
        #     core_bsn_query_prompt_tmpl.format(
        #         company_name=ev.company_name
        #     )).response.strip()

        retr_output = "I don't know."

        if retr_output == "I don't know.":
            return _UrlSearchEvent(query=ev.company_name)

        return _StopEvent(query=retr_output)

    @step()
    async def get_web_url(
            self,
            ev: _UrlSearchEvent,
            ctx: Context
    ) -> _UrlContentExtractionEvent:
        """aaa"""

        print(ev.query)
        # web_search_output = DDGS(verify=False) \
        #     .text(keywords=f"What does {ev.query} do?",
        #           max_results=2)
        #
        # with open('./tmp_ddg_ferrari.pkl', 'wb') as f:
        #     pickle.dump(web_search_output, f)

        with open('./tmp_ddg_ferrari.pkl', 'rb') as f:
            web_search_output = pickle.load(f)

        await ctx.set(key="Web search output", value=web_search_output)

        # query = ev.query
        # web_search_query = llm \
        #     .as_structured_llm(WebSearchQueryDefinitionFormat)\
        #     .complete(web_search_query_prompt_tmpl.format(question_str=query))\
        #     .raw
        #
        # print(f'question: {web_search_query.question}')
        # print(f'query: {web_search_query.query}')

        return _UrlContentExtractionEvent(query=ev.query)

    @step()
    async def get_web_contents(
            self,
            ev: _UrlContentExtractionEvent,
            ctx: Context
    ) -> _StopEvent:
        """aaa"""

        print(ev.query)
        web_search_output = await ctx.get("Web search output")
        print(web_search_output)
        # web_documents = web_content_loader\
        #     .load_data(urls=[el['href'] for el in web_search_output_prc])
        #
        # with open('./tmp_bs_ferrari.pkl', 'wb') as f:
        #     pickle.dump(web_documents, f)
        with open('./tmp_bs_ferrari.pkl', 'rb') as f:
            web_documents = pickle.load(f)

        # Put full body context in web_documents
        web_documents_prc = copy.deepcopy(web_documents)
        for i, web_document in enumerate(web_documents):
            web_documents_prc[i].text = re.sub(r'\n+', r' ', web_document)
            # Da fare, inserire anche il body in web document e assicurarsi che
            # sia ben letto da chromadb e dal llm

        await ctx.set(key="Web scraping output", value=web_documents)
    #     query = ev.query
    #     web_search_query = llm_text_generation \
    #         .as_structured_llm(WebSearchQueryDefinitionFormat) \
    #         .complete(web_search_query_prompt_tmpl.format(question_str=query)) \
    #         .raw
    #
    #     print(f'question: {web_search_query.question}')
    #     print(f'query: {web_search_query.query}')
    #
        return _StopEvent(query="Exit - correct")


async def web_search_workflow_execution():
    """Main function"""
    w = WebSearchWorkflow(timeout=10, verbose=False)
    response = await w.run(company_name="Ferrari")
    print(response)
