from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
from typing import Optional
import re

from conf import config

class Tariffs(BaseModel):
    light_dues: Optional[str] = Field(description="Light Dues for the vessel")
    port_dues: Optional[str] = Field(description="Port Dues for the vessel")
    towage_dues: Optional[str] = Field(description="Towage Dues for the vessel")
    vts_dues: Optional[str] = Field(description="Vehicle Traffic Services (VTS) dues for the vessel")
    pilotage_dues: Optional[str] = Field(description="Pilotage Dues for the vessel")
    running_of_vessel_lines_dues: Optional[str] = Field(description="Running of Vessel Lines Dues for the vessel")


llm = GoogleGenAI(
    api_key=config.google_genai_api_key,
    model=config.google_genai_model_better
)
llm = llm.as_structured_llm(Tariffs)

# embeddings model settings, use whichever
embed_model = HuggingFaceEmbedding(
    # model_name="sentence-transformers/all-MiniLM-L6-v2"     # free, lightweight, high-performing model
    # model_name="BAAI/bge-small-en"                        # free, open-source
    model_name="sentence-transformers/all-mpnet-base-v2"  # free, better accuracy, larger, slower
)

# apply settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512       # chunks of 512 tokens
Settings.chunk_overlap = 50     # Overlap to retain context


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
    
    # async def _generate_better_query(self, original_query):
    #     # Use LLM to generate a more targeted query
    #     prompt = f"""
    #     I want to search for information in a port tariff document. 
    #     Given the following user query, generate 2-3 specific search queries that would help find the exact tariff information needed:
        
    #     Original query: {original_query}
        
    #     Format your response as JSON with a list of queries. Example:
    #     {{
    #         "queries": [
    #             "specific search query 1",
    #             "specific search query 2"
    #         ]
    #     }}
    #     """
        
    #     response = await llm.acomplete(prompt)
    #     try:
    #         query_data = json.loads(response.text)
    #         return query_data["queries"]
    #     except:
    #         # Fallback to original query if parsing fails
    #         return [original_query]
    
    async def _get_query_engine(self, documents):
        # Create transformer for better chunking
        text_splitter = TokenTextSplitter(
            chunk_size=512,  # Smaller chunks for more precise retrieval
            chunk_overlap=50
        )
        
        # Use ingestion pipeline with better transformations
        pipeline = IngestionPipeline(
            transformations=[text_splitter, embed_model]
        )
        
        nodes = await pipeline.arun(documents=documents, num_workers=3)
        
        # Use vector store with metadata filtering capabilities
        index = VectorStoreIndex(nodes=nodes)
        
        # Reranker to prioritize most relevant chunks. reduces noise and improves accuracy
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-TinyBERT-L-2-v2", 
            # model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            top_n=10  # Retrieve more candidates for reranking
        )
        
        # Create query engine with better parameters
        query_engine = index.as_query_engine(
            similarity_top_k=30,                # Retrieve more initially
            node_postprocessors=[reranker],
            response_mode="compact"             # for factual retrieval, another option=tree_summarize
        )
        
        return query_engine
    
    
    async def query_llm(self, query: str):
        # create page-wise documents
        documents = await self._get_documents()

        # create query-engine over documents
        query_engine = await self._get_query_engine(documents=documents)

        # # Generate better queries
        # better_queries = await self._generate_better_query(query)
        
        # # Run multiple queries and combine results
        # all_responses = []
        # all_sources = []
        
        # for better_query in better_queries:
        #     response = await query_engine.aquery(better_query)
        #     all_responses.append(response.response)
        #     all_sources.extend(response.source_nodes)
        
        # # Use LLM to synthesize the final answer
        # synthesis_prompt = f"""
        # Given the following information retrieved from a port tariff document, calculate the tariffs for the vessel described in the original query:
        
        # Original query: {query}
        
        # Retrieved information:
        # {' '.join(all_responses)}
        
        # Respond with a structured breakdown of all applicable tariffs.
        # """
        # final_response = await llm.acomplete(synthesis_prompt)

        # query over documents
        response = await query_engine.aquery(query)
        results = {
            "answer": response.response,
            "sources": [
                {"filename": node.metadata["filename"], "section": node.metadata["section"]}
                for node in response.source_nodes
            ]
        }

        return results
