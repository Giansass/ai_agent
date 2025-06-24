"""aaa"""
import re
import time

import chromadb
import tiktoken
import validators
from duckduckgo_search import DDGS
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
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
from src.utils.bing_search_loader import bing_search
from src.utils.custom_dataclass import WebSearchConfigData, WebSearchQueryData
from src.utils.env_var_loader import (
    BING_FROM_DATE,
    BING_HEADERS,
    BING_N_PAGES,
    BING_SEARCH_URL,
    BING_SIZE_PAGE,
    BING_TO_DATE,
    DUCKDUCKGO_MAX_RESULTS,
    PHOENIX_COLLECTOR_ENDPOINT,
    TOKEN_SPLITTER_CHUNK_OVERLAP,
    TOKEN_SPLITTER_CHUNK_SIZE,
    TOKEN_SPLITTER_MODEL_NAME,
    WEB_SEARCH_ENGINE,
    WORKFLOW_TIMEOUT,
)
from src.utils.prompts import (
    CORE_BSN_QUERY_PROMPT_TEMPLATE_STR,
    WEB_CRAWLING_QUERY_PROMPT_STR,
)
from src.utils.web_contents_validator import web_contents_validator

# Set phoenix observability monitor
tracer_provider = register(
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    project_name="Web search agent")

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Create prompts
core_bsn_query_prompt_tmpl = RichPromptTemplate(CORE_BSN_QUERY_PROMPT_TEMPLATE_STR)
# web_search_query_prompt_tmpl = RichPromptTemplate(WEB_SEARCH_QUERY_PROMPT_STR)
web_crawling_query_prompt_tmpl = RichPromptTemplate(WEB_CRAWLING_QUERY_PROMPT_STR)

# Web content loader
web_content_loader = BeautifulSoupWebReader()

# Sentence and text splitter parser
parser = SentenceSplitter()
text_splitter = TokenTextSplitter(chunk_size=TOKEN_SPLITTER_CHUNK_SIZE,
                                  chunk_overlap=TOKEN_SPLITTER_CHUNK_OVERLAP)

# Token counter
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.get_encoding(TOKEN_SPLITTER_MODEL_NAME).encode
)

Settings.callback_manager = CallbackManager([token_counter])
token_counter.reset_counts()


# Event definitions
class _StartEvent(StartEvent):
    """Custom start event

    Parameters
    ----------
    query : str
    first_try: bool, default True

    """

    query: str
    first_try: bool = True


# Event definitions
class _QueryEvent(Event):
    """Event used to

    Parameters
    ----------
    query : str
    first_try: bool, default False

    """

    query: str
    first_try: bool = False


class _StopEvent(StopEvent):
    """Custom stop event

    Parameters
    ----------
    query : str

    """

    query: str


class _UrlSearchEvent(Event):
    """Event used to search web contents through the
    web search engine.

    Parameters
    ----------
    query : str

    """

    query: str


class _UrlContentExtractionEvent(Event):
    """Event used to extract web contents.

    Parameters
    ----------
    query : str

    """

    query: str


class _UrlContentStoringEvent(Event):
    """Event used to store web contents.

    Parameters
    ----------
    query : str

    """

    query: str


# Pydantic data structures
class CompanySearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output

        Parameters
        ----------
        obj : type

        Returns
        -------
        obj : type
            description
        """

    company_name: str = Field(description="The company name")
    company_core_business: str = Field(description="The company core business")


pydantic_output_parser = PydanticOutputParser(
    output_cls=CompanySearchQueryDefinitionFormat
)


class WebSearchQueryDefinitionFormat(BaseModel):
    """description

    Parameters
    ----------
    obj : type

    Returns
    -------
    obj : type
    """

    question: str = Field(description="The question")
    query: str = Field(description="The search engine query")


class TextCleaner(TransformComponent):
    """description

    Parameters
    ----------
    obj : type

    Returns
    -------
    obj : type
    """
    def __call__(self, nodes, **kwargs):
        prc_nodes = []
        for node in nodes:
            prc_text = re.sub(r"\n+", r" ", node.text)
            prc_text = re.sub(r"\r+", r" ", prc_text)
            prc_text = re.sub(r"\t+", r" ", prc_text)
            prc_text = re.sub(r"\s+", r" ", prc_text)
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
    similarity_top_k=1,
    vector_store_query_mode="default",
    llm=llm_text_generation,
    output_parser=pydantic_output_parser
)


class WebSearchWorkflow(Workflow):
    """Class used to define the agent workflow"""

    @step()
    async def get_core_business_from_index(
            self,
            ev: _StartEvent | _QueryEvent,

    ) -> _UrlSearchEvent | _StopEvent:

        """Very first step of the workflow. The agent try to answer
        using the available contents. If contents already ingested
        are not enough return _QueryEvent, otherwise _StopEvent.

        Parameters
        ----------
        ev : _StartEvent | _QueryEvent

        Returns
        -------
        _UrlSearchEvent | _StopEvent
            Return _UrlSearchEvent if no contents found.
            _StopEvent otherwise.
        """

        retrieved_output = query_engine.query(
                                core_bsn_query_prompt_tmpl.format(
                                    company_name=ev.query
                                    )
                                ).response.strip()  # type: ignore

        n_tokens = f"""
        Embedding Tokens: {token_counter.total_embedding_token_count}
        LLM Prompt Tokens: {token_counter.prompt_llm_token_count}
        LLM Completion Tokens: {token_counter.completion_llm_token_count}
        Total LLM Token Count: {token_counter.total_llm_token_count}
        """
        print(n_tokens)

        bad_resp = ["Empty Response", "I don't know."]
        if retrieved_output in bad_resp and ev.first_try:
            return _UrlSearchEvent(query=ev.query)

        return _StopEvent(query=retrieved_output)

    @step()
    async def get_web_url(
            self,
            ev: _UrlSearchEvent,
            ctx: Context) -> _UrlContentExtractionEvent | _StopEvent:
        """This step is in charge to call Bing Search API to retrieve urls
        that contains info about the asked company. If no new contents have
        been found the method returns _StopEvent. Otherwise _UrlContentExtractionEvent

        Parameters
        ----------
        ev : _UrlSearchEvent
        ctx : Context

        Returns
        -------
        _UrlContentExtractionEvent | _StopEvent
        """

        web_search_query = WebSearchQueryData(
            company=ev.query,
            from_date=BING_FROM_DATE,
            to_date=BING_TO_DATE
        )

        web_search_query_config = WebSearchConfigData(
            size_page=BING_SIZE_PAGE,
            n_pages=BING_N_PAGES,
            search_url=BING_SEARCH_URL,
            headers=BING_HEADERS
        )

        if WEB_SEARCH_ENGINE == "DuckDuckGo":

            # import os
            # print(os.environ.get("REQUESTS_CA_BUNDLE"))
            # print(os.environ.get("SSL_CERT_FILE"))
            # print('aaa')
            # xfvxff

            # DuckDuckGo Search API
            start_time = time.time()
            print(f"Web crawling started for {web_search_query.company}...")

            # Prepare the query
            duckduck_go_query = web_crawling_query_prompt_tmpl.format(
                                    company_name=web_search_query.company
                                ).strip()

            # Retrieve the content
            print(f"Querying DuckDuckGo with: {duckduck_go_query}")
            retrieved_content_temp = DDGS().text(
                                                keywords=duckduck_go_query,
                                                max_results=DUCKDUCKGO_MAX_RESULTS)
            print(f"Web crawling took {time.time() - start_time:.2f} seconds.")

            # Replace href with url
            retrieved_content = []
            for cont in retrieved_content_temp:
                cont['url'] = cont['href']
                del cont['href']
                retrieved_content.append(cont)

            print(f"Retrieved {len(retrieved_content)} results from DuckDuckGo.")
            del retrieved_content_temp

        elif WEB_SEARCH_ENGINE == "Bing":
            # Bing Search API
            retrieved_content = bing_search(web_search_query, web_search_query_config)

        else:
            raise ValueError(f"Web search engine {WEB_SEARCH_ENGINE} not supported.")

        # Validate retrieved content
        verified, query = web_contents_validator(retrieved_content)

        # If no contents have been found
        if not verified:
            return _StopEvent(query=query)

        urls = []
        for el in retrieved_content:
            # Check if the retrieved url is valid
            if validators.url(el['url']):
                print(f'Valid URL found: {el["url"]}')
                urls.append(el['url'])
            # If the url is not valid, print a warning
            else:
                print(f'Invalid URL found: {el["url"]}')
        print(f"Retrieved {len(urls)} valid URLs.")

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
        """Once Web Search API has been used to retrieve urls this method is in
        charge to scrape urls obtaining the full body contents.

        Parameters
        ----------
        ev : _UrlContentExtractionEvent
        ctx : Context

        Returns
        -------
        _UrlContentStoringEvent
        """

        urls_to_prc = await ctx.get("urls to prc")

        start_time = time.time()
        web_documents = web_content_loader\
            .load_data(urls=urls_to_prc)
        end_time = time.time()
        print(f"Web scraping took {end_time - start_time:.2f} seconds.")

        start_time = time.time()
        pipeline = IngestionPipeline(
            transformations=[
                TextCleaner(),
                text_splitter,
                llm_text_embedding  # type: ignore
            ]
        )
        end_time = time.time()
        print(f"Pipeline setup took {end_time - start_time:.2f} seconds.")

        print(3)
        # run the pipeline
        nodes = pipeline.run(documents=web_documents, show_progress=True)
        # Da fare, inserire anche il body in web document e assicurarsi che
        # sia ben letto da chromadb e dal llm

        await ctx.set(key="nodes", value=nodes)
        return _UrlContentStoringEvent(query=ev.query)

    @step()
    async def store_web_contents(
            self,
            ev: _UrlContentStoringEvent,
            ctx: Context
    ) -> _QueryEvent:
        """The method is in charge to store new retrieved web contents on chromadb.
        once completed the workflow come back to get_core_business_from_index.

        Parameters
        ----------
        ev : _UrlContentStoringEvent
        ctx : Context

        Returns
        -------
        _QueryEvent
        """

        nodes = await ctx.get("nodes")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # save to disk
        _ = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=llm_text_embedding
        )

        return _QueryEvent(query=ev.query)


async def web_search_workflow_execution():
    """The function is intended to execute the workflow through the __main__ script
    and print the results."""
    w = WebSearchWorkflow(timeout=WORKFLOW_TIMEOUT, verbose=True)
    response = await w.run(query="Vodafone", first_try=True)
    print(response)
