"""aaa"""
import re

import chromadb
from duckduckgo_search import DDGS
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import TransformComponent
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

#
parser = SentenceSplitter()

text_splitter = TokenTextSplitter(chunk_size=512)


# Event definitions
class _StartEvent(StartEvent):
    """aaa"""

    query: str


# Event definitions
class _QueryEvent(Event):
    """aaa"""

    query: str
    first_try: bool = True


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


class TextCleaner(TransformComponent):
    """aaa"""
    def __call__(self, nodes, **kwargs):
        prc_nodes = []
        for node in nodes:
            prc_text = re.sub(r"\n+", r" ", node.text)
            prc_nodes.append(
                Document(id=node.id_,
                         text=prc_text,
                         metadata={"href": node.id_}))
        return prc_nodes


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
            ev: _StartEvent | _QueryEvent,

    ) -> _UrlSearchEvent | _StopEvent:

        """aaa"""

        retrieved_output = query_engine.query(
                                core_bsn_query_prompt_tmpl.format(
                                    company_name=ev.query
                                )).response.strip()

        bad_resp = ["Empty Response", "I don't know."]
        if retrieved_output in bad_resp and ev.first_try:
            return _UrlSearchEvent(query=ev.query)

        return _StopEvent(query=retrieved_output)

    @step()
    async def get_web_url(
            self,
            ev: _UrlSearchEvent,
            ctx: Context
    ) -> _UrlContentExtractionEvent | _StopEvent:
        """aaa"""

        web_search_output = DDGS(verify=False).text(
                                keywords=f"What does {ev.query} do?",
                                max_results=5)

        # with open('./tmp_ddg_ferrari.pkl', 'wb') as f:
        #     pickle.dump(web_search_output, f)
        #
        # with open('./tmp_ddg_ferrari.pkl', 'rb') as f:
        #     web_search_output = pickle.load(f)

        # Check if web contents already exists
        urls = [web_doc['href'] for web_doc in web_search_output]
        urls_to_prc = []
        for url in urls:
            collected = chroma_collection.get(where={"href": url})
            if len(collected['ids']) == 0:
                urls_to_prc.append(url)

        if len(urls_to_prc) == 0:
            query = "No other contents found. Not able to answer the question"
            return _StopEvent(query=query)

        await ctx.set(key="urls to prc", value=urls_to_prc)

        return _UrlContentExtractionEvent(query=ev.query)

    @step()
    async def get_web_contents(
            self,
            ev: _UrlContentExtractionEvent,
            ctx: Context
    ) -> _UrlContentStoringEvent:
        """aaa"""

        urls_to_prc = await ctx.get("urls to prc")
        web_documents = web_content_loader\
            .load_data(urls=urls_to_prc)

        pipeline = IngestionPipeline(
            transformations=[
                TextCleaner(),
                text_splitter,
                llm_text_embedding
            ]
        )

        # run the pipeline
        nodes = pipeline.run(documents=web_documents)

        # index.update_ref_doc(doc_chunks[0])
        # Da fare, inserire anche il body in web document e assicurarsi che
        # sia ben letto da chromadb e dal llm

        await ctx.set(key="nodes", value=nodes)
    #     query = ev.query
    #     web_search_query = llm_text_generation \
    #         .as_structured_llm(WebSearchQueryDefinitionFormat) \
    #         .complete(web_search_query_prompt_tmpl.format(question_str=query)) \
    #         .raw
    #
    #     print(f'question: {web_search_query.question}')
    #     print(f'query: {web_search_query.query}')
    #
        return _UrlContentStoringEvent(query=ev.query)

    @step()
    async def store_web_contents(
            self,
            ev: _UrlContentStoringEvent,
            ctx: Context
    ) -> _QueryEvent:
        """aaa"""

        nodes = await ctx.get("nodes")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # save to disk
        _ = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=llm_text_embedding
        )

        return _QueryEvent(query=ev.query, first_try=False)


async def web_search_workflow_execution():
    """Main function"""
    w = WebSearchWorkflow(timeout=10, verbose=False)
    response = await w.run(query="KPMG")
    print(response)
