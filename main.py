import asyncio
import time

from model.agent_document_loader import DocumentLoader
from model.agent_porttariff_engine import PortTariffEngine
from model.agent_google_docintel import GoogleGenAIDocIntel
from query import user_query

if __name__ == "__main__":
    time_start = time.time()

    # Run Agent1: DocumentLoader to get doc text. (Note: Due to free-tier usage, all concurrency & parallelization is removed. You may add if needed.)
    doc_loader = DocumentLoader(pdf_path="./data/Port Tariff.pdf")      # Enter path of tariff document.pdf here
    # text = doc_loader.process_document()
    with open("./data/Port Tariff.txt", "r") as f:                      # Enter path of tariff document.txt here
        text = f.read()                         # as process_ducument runs sequentially for this implementation, using locally saved data to save time

    # Run Agent2: PortTariffEngine to query over text.
    tariff_engine = PortTariffEngine(filename="Port Tariff.txt", text=text)
    response = asyncio.run(tariff_engine.query_llm(query=user_query))

    print("Response_RAG:", response['answer'], end="\n")

    docintel = GoogleGenAIDocIntel(file_path="./data/Port Tariff.pdf")
    response_docintel = docintel.get_response(user_query)
    print("\nResponse_GoogleGenAIDocIntel:", response_docintel.text, end="\n")
    print(f"Time Taken: {time.time()-time_start}")
