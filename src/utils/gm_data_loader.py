"""The module is intended to ingest gm data to chroma vector store"""

import os
from typing import Optional

import chromadb
import pandas as pd
import tiktoken
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.llama_index_experiments.llm_load import llm_text_embedding, llm_text_generation
from src.utils.env_var_loader import TOKEN_SPLITTER_MODEL_NAME
from src.utils.prompts import FIND_SIMILAR_MG_QUERY_PROMPT_STR

SIMILAR_MGS_QUERY_PROMPT_TMPL = RichPromptTemplate(FIND_SIMILAR_MG_QUERY_PROMPT_STR)
INGEST_DATA = False
COMPANY_DESCRIPTION = """
    SOMI Impianti S.r.l. offers comprehensive solutions for the Oil and Gas industry.
    It is a leading turnkey power plant developer, providing industrial solutions for
    the installation of various equipment. The company also offers a wide range of
    oilfield products and manufacturing services, including the manufacturing and
    erection of oil and gas pipes. Additionally, SOMI provides industrial demolition
    services.
    """


class MGSearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output"""

    output: list[list[str]] = Field(
        description="List of most similar MGs to the company description.",
        example=[
            """
            [
                ["Red products", "Red products are a key focus for this company."],
                ["Red services", "The company has a strong presence in the red market."]
            ]
            """,
        ])


class GMDataLoader:
    """Class to load Gemini data from environment variables."""

    def __init__(self,
                 chromadb_path: str,
                 chromadb_collection_name: str,
                 llm_emb,
                 llm_gen,
                 top_k: int = 10,
                 ):

        # Initialize the GMDataLoader with parameters
        self.chromadb_path = chromadb_path
        self.chromadb_collection_name = chromadb_collection_name
        self.llm_emb = llm_emb
        self.llm_gen = llm_gen
        self.top_k = top_k

        # Initiliaze the llm output parser
        self.output_parser = PydanticOutputParser(
            output_cls=MGSearchQueryDefinitionFormat
        )

        # Initialize ChromaDB client
        db2 = chromadb.PersistentClient(path=self.chromadb_path)

        # Create or get the collection
        chroma_collection = db2.get_or_create_collection(self.chromadb_collection_name)

        # Create the vector store using the Chroma collection
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create the index from the vector store
        self.index = VectorStoreIndex.from_vector_store(
                        self.vector_store,
                        embed_model=self.llm_emb
                        )

        # Create the query engine with the index and output parser
        self.query_engine = self.index.as_query_engine(
                                similarity_top_k=self.top_k,
                                vector_store_query_mode="default",
                                llm=self.llm_gen,
                                output_parser=self.output_parser
                            )

        # Create a retriever from the index
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

        # Initialize the token counter
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.get_encoding(TOKEN_SPLITTER_MODEL_NAME).encode
        )

        Settings.callback_manager = CallbackManager([self.token_counter])
        self.token_counter.reset_counts()

    def _extract_gm_from_excel(self,
                               input_file):
        """Extracts MG data from an Excel file."""

        print("Extracting MG data from Excel file:", input_file)
        df = pd.read_excel(input_file,
                           engine='openpyxl',
                           header=1,
                           usecols="A:K",
                           dtype=str)

        # Rename columns for easier access
        df.rename(columns={
                    'Ambito - Codice': 'scope_code',
                    'Ambito (Italian)': 'scope_italian',
                    'Famiglie - Codice': 'family_code',
                    'Famiglie (Italian)': 'family_italian',
                    'Classi - Codice': 'class_code',
                    'Classi (Italian)': 'class_italian',
                    'MG Code': 'mg_id',
                    'MG Desc (Italian)': 'mg_italian',
                    'MG Desc (English)': 'mg_english'
                }, inplace=True)

        print(f"Extracted {len(df)} rows from the Excel file.")
        assert df['mg_id'].nunique() == len(df), "MG IDs are not unique."

        nr_mg_nan = df['mg_english'].isna().sum()
        err_msg = "Found NaN values in MG English descriptions, expected 0."
        assert nr_mg_nan == 0, err_msg

        nr_mg_empty = df['mg_english'].eq('').sum()
        err_msg = "Found Empty values in MG English descriptions, expected 0."
        assert nr_mg_empty == 0, err_msg

        print("MG data extraction completed successfully.")

        return df

    def _load_gm_data_in_chromadb(self, df: pd.DataFrame):
        """Loads gm data into ChromaDB."""

        # Initialize ChromaDB client
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Convert DataFrame to Document nodes
        mg_english_list = df[['mg_english', 'mg_id']].to_dict(orient='records')

        mg_nodes = []
        for mg in mg_english_list:
            if not mg['mg_english'] or pd.isna(mg['mg_english']):
                raise ValueError(f"Invalid MG English description for ID {mg['mg_id']}")

            # Ensure that mg['mg_english'] is a string
            if not isinstance(mg['mg_english'], str):
                err = f"MG English description for ID {mg['mg_id']} is not a string."
                raise TypeError(err)

            # Create a TextNode for each MG description
            mg_nodes.append(TextNode(text=mg['mg_english'], id_=mg['mg_id']))

        print(f"Converted {len(mg_nodes)} MG descriptions to TextNodes.")

        # save to disk
        _ = VectorStoreIndex(
            mg_nodes,
            storage_context=storage_context,
            embed_model=self.llm_emb
        )
        print("MG data loaded into ChromaDB successfully.")

    def find_most_similar(self, company_description: str):
        """Finds the most similar documents to the query."""

        company_description = company_description.strip()
        if not company_description:
            raise ValueError("Company description cannot be empty.")

        # Find most similar documents
        query = SIMILAR_MGS_QUERY_PROMPT_TMPL.format(
                    company_description=company_description
                ).strip()

        # Identify similar MG nodes using the retriever
        similar_nodes = self.retriever.retrieve(query)
        print(f"\t Retrieved {len(similar_nodes)} similar MG nodes.")

        print("\t Similar MGs used by the LLM:")
        for node in similar_nodes:

            text = node.text.strip()
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"\t> MG ID: {node.id_}, Text: {text}, Score: {node.score:.3f}")
        print('\n')

        # Retrieve output using the query engine
        print("\t Retrieving output using the query engine...")
        answer = self.query_engine.query(query).response.strip()
        print("\t Output retrieved successfully.")

        n_tokens = f"""
        > Embedding Tokens: {self.token_counter.total_embedding_token_count}
        > LLM Prompt Tokens: {self.token_counter.prompt_llm_token_count}
        > LLM Completion Tokens: {self.token_counter.completion_llm_token_count}
        > Total LLM Token Count: {self.token_counter.total_llm_token_count}
        """
        print(n_tokens)

        return similar_nodes, answer

    def _validate_answer(self, output: str):
        """Validates the output from the LLM against the expected format."""
        try:
            parsed_output = self.output_parser.parse(output)
            return parsed_output.output
        except Exception as e:
            raise ValueError(f"Output validation failed: {e}") from e

    def _output_definition(self,
                           cos_sim_mgs: list[TextNode],
                           answer: dict) -> list[tuple[str, str, str]]:
        """Formats the output definition for the similar MGs."""

        mgs_code_list = []
        for mg in cos_sim_mgs:
            mgs_code_list.append((mg.id_, mg.text.strip()))

        answer_list = []
        for mg, motivation in answer:
            # Find the corresponding MG code in the cos_sim_mgs
            mg_code = next((code for code, text in mgs_code_list
                            if text == mg.strip()), None)
            if mg_code:
                answer_list.append((mg_code, mg, motivation.strip()))

        return answer_list

    def main(self,
             company_description: str,
             ingest_data: bool = False,
             input_file_path: Optional[str] = None,
             input_file_name: Optional[str] = None):
        """Main method to run the GMDataLoader."""

        if ingest_data:
            if not input_file_path or not input_file_name:
                raise ValueError("Input file path and name must be provided.")
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"{input_file_path} does not exist.")

            # Check if the input file path and name are provided
            input_file = os.path.join(input_file_path, input_file_name)

            print("- Loading GM data: START")
            if not os.path.exists(input_file):
                print("- Loading GM data: ERROR")
                raise FileNotFoundError(f"{input_file} does not exist.")

            gm_df = self._extract_gm_from_excel(input_file)
            print("- Loading GM data: OK\n")

            print("- Ingesting GM data into ChromaDB: START")
            self._load_gm_data_in_chromadb(gm_df)
            print("- Ingesting GM data into ChromaDB: OK\n")

        print("- Finding most similar MGs for the company description: START")
        cos_sim_mgs, answer = self.find_most_similar(company_description)
        print("- Finding most similar MGs for the company description: OK\n")

        print("- Validating the output from the LLM: START")
        answer = self._validate_answer(answer)
        if not answer:
            raise ValueError("No similar MGs found for the given company description.")
        print("- Validating the output from the LLM: OK\n")

        print("- Formatting the output definition for the similar MGs: START")
        if not cos_sim_mgs:
            raise ValueError("No similar MGs found in the ChromaDB.")

        answer = self._output_definition(cos_sim_mgs, answer)
        print("- Formatting the output definition for the similar MGs: OK\n")

        print("\t Similar MGs found:\n")
        for mg in answer:
            mg_id, mg_desc, motivation = mg
            print(f"\t> MG: {mg_id} - {mg_desc} > Motivation: {motivation}")
        print("\t> All similar MGs found successfully.\n")

        return answer


if __name__ == "__main__":

    # Create the GMDataLoader instance
    print("Initializing GMDataLoader...\n")
    gm_loader = GMDataLoader(
                    chromadb_path=".storage/chroma/gm_descriptions",
                    chromadb_collection_name="gm_definition_collection",
                    llm_emb=llm_text_embedding,
                    llm_gen=llm_text_generation,
                )
    # Run the main method with the company description
    print("Running GMDataLoader main method: START")
    gm_loader.main(COMPANY_DESCRIPTION)
    print("Running GMDataLoader main method: OK")
