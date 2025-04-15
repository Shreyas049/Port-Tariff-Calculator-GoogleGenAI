from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, MarkdownNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional
import re
import json

from conf import config
from query import vessel_details


class Tariffs(BaseModel):
    """<Think>Extract pydantic information below"""
    light_dues: Optional[str] = Field(description="Light dues required by the vessel for berting at given port.")


class ModelSettings():
    system_prompt = """
    **You are an expert at reading Tariff documents for ships/vessels at variousPorts.**
    **Based on the provided Vessel Details, your job is to think carefully and answer query accurately.**
    **For any response if any calculations are needed, think, calculate and only then answer.**
    **Do not provide any explanations or additional information in response. Only answer whatever is asked.**\n
    **Try to provide answer in JSON format. For e.g. [{'key_1': 'value_1', 'key_2', 'value_2'}]**
    """

    llm = GoogleGenAI(
        model=config.google_genai_model_better,
        api_key=config.google_genai_api_key,
        generation_config=types.GenerateContentConfig(
            system_instruction=f"""{system_prompt}\n{vessel_details}""",
            # response_mime_type='application/json',
            # response_schema=Tariffs,
        )
    )
    # llm = llm.as_structured_llm(Tariffs)

    # embeddings model settings, use whichever
    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        api_key=config.google_genai_api_key
    )
    
    # embed_model = HuggingFaceEmbedding(
    #     # model_name="sentence-transformers/all-MiniLM-L6-v2"     # free, lightweight, high-performing model
    #     # model_name="BAAI/bge-small-en"                        # free, open-source
    #     # model_name="sentence-transformers/all-mpnet-base-v2"  # free, better accuracy, larger, slower
    # )


class PortTariffEngine:
    def __init__(self, filename: str, text: str):
        self.filename = filename
        self.text = text

    async def _get_documents(self):
        # Define the regex pattern for splitting
        # Matches "**SECTION <digit>**"
        pattern = r'(\*\*SECTION\s\d+\*\*)'
        
        # Split the text based on the pattern
        # re.split keeps the delimiters in the result, so we filter them out if needed
        sections = re.split(pattern, self.text)
        
        # Remove empty strings and strip whitespace
        sections = [s.strip() for s in sections if s.strip() and not re.match(pattern, s.strip())]
        
        tariff_doc = {}

        for i, section in enumerate(sections):
            tariff_doc[f"Section {i+1}"] = section

        documents = []
        for section, content in tariff_doc.items():
            documents.append(
                Document(text=content, metadata={"filename": self.filename, "section": section})
            )
        
        return documents
        
    async def _get_query_engine(self, text):
        # Create documents
        documents = [Document(text=text, metadata={"filename": self.filename})]
        
        # Create transformer for better chunking
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[4096, 2048, 1024, 512],
            chunk_overlap=256
        )
        nodes = parser.get_nodes_from_documents(documents)

        # per-index
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=ModelSettings.embed_model
        )

        # # Reranker to prioritize most relevant chunks. reduces noise and improves accuracy
        # reranker = SentenceTransformerRerank(
        #     model="cross-encoder/ms-marco-TinyBERT-L-2-v2", 
        #     # model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        #     top_n=10        # Retrieve more candidates for reranking
        # )
        
        # Create query engine with better parameters
        query_engine = index.as_query_engine(
            llm=ModelSettings.llm,
            similarity_top_k=20,                # Retrieve more initially
            response_mode="compact"             # for factual retrieval, another option=tree_summarize
            # node_postprocessors=[reranker],
        )
        return query_engine
    
    
    async def query_llm(self, query: str):
        # create query-engine over documents
        query_engine = await self._get_query_engine(text=self.text)

        # query over documents
        response = await query_engine.aquery(query)
        results = {
            "answer": response.response,
            "sources": [
                {"filename": node.metadata["filename"]}
                for node in response.source_nodes
            ]
        }

        return results
