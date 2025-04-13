from google import genai
from google.genai import types
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from conf import config

# from concurrent.futures import ProcessPoolExecutor
# import functools

llm_api_key = config.google_genai_api_key
llm_model = config.google_genai_model_smaller


class DocumentLoader():
    def __init__(self, pdf_path: str):
        # Open PDF and get the page
        self.doc = fitz.open(pdf_path)

        # Create google-genai client
        self.client = genai.Client(api_key=llm_api_key)

    RETRY_DECORATOR = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=10)
    )
    
    @RETRY_DECORATOR
    def call_gemini(self, image):
        """Synchronous version of call_gemini"""
        image = Image.open(image)
        response = self.client.models.generate_content(
            model=llm_model,
            contents=[
                image,
                "Read the image carefully and extract all the information in text. Represent Section, SubSection, Tables in text output whereever necessary."
            ]
        )
        return response.text
    
    def process_page(self, page_num):
        # Get the page
        page = self.doc[page_num]
        # Get page dimensions
        width, height = page.rect.width, page.rect.height
        
        # Extract left half
        left_rect = fitz.Rect(0, 0, width/2, height)
        left_img = page.get_pixmap(clip=left_rect)
        left_buffer = BytesIO(left_img.tobytes("png"))
        left_buffer.seek(0)
        
        # Extract right half
        right_rect = fitz.Rect(width/2, 0, width, height)
        right_img = page.get_pixmap(clip=right_rect)
        right_buffer = BytesIO(right_img.tobytes("png"))
        right_buffer.seek(0)
        
        # Use synchronous version of call_gemini
        left_response = doc_loader.call_gemini(left_buffer)
        right_response = doc_loader.call_gemini(right_buffer)
        
        actual_page = 1 + 2 * page_num
        text = f"\n--- Page {actual_page} ---\n{left_response}\n\n--- Page {actual_page+1} ---\n{right_response}\n\n"
        return text
    
    def process_document(self):
        results = ""
        for page_no in range(len(self.doc)):
            start_time = time.time()
            response = self.process_page(page_num=page_no)
            results += response
            print(f"Processed page: {page_no}, time taken={time.time()-start_time}")

        return results
    

if __name__ == "__main__":
    print("Initiating DocumentLoader")
    start_time = time.time()

    doc_loader = DocumentLoader(pdf_path="./Port Tariff.pdf")
    text = doc_loader.process_document()
    with open("./Port Tariff.txt", "w") as f:
        f.write(text)

    print("Processing Complete!")
    print(f"Total Time Taken DocumentLoader: {time.time() - start_time}")
