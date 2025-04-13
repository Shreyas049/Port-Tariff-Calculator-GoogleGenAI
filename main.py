import asyncio

from model.agent_document_loader import DocumentLoader
from model.agent_porttariff_engine import PortTariffEngine
from prompts import user_prompt

if __name__ == "__main__":
    # Run Agent1: DocumentLoader to get doc text. (Note: Due to free-tier usage, all concurrency & parallelization is removed. You may add if needed.)
    doc_loader = DocumentLoader(pdf_path="./data/Port Tariff.pdf")      # Enter path of tariff document.pdf here
    # text = doc_loader.process_document()
    with open("./data/Port Tariff.txt", "r") as f:                      # Enter path of tariff document.txt here
        text = f.read()                         # as process_ducument runs sequentially for this implementation, using locally saved data to save time

    # Run Agent2: PortTariffEngine to query over text.
    tariff_engine = PortTariffEngine(filename="Port Tariff.txt", text=text)
    response = asyncio.run(tariff_engine.query_llm(query=user_prompt))

    print(f"{response=}")
