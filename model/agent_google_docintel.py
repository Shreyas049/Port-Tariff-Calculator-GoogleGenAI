from google import genai
from google.genai import types
import pathlib

from conf import config
from query import vessel_details


system_prompt = """
**You are an expert at reading Tariff documents for ships/vessels at variousPorts.**
**Based on the provided Vessel Details, your job is to think carefully and answer query accurately.**
**For any response if any calculations are needed, think, calculate and only then answer.**
**Do not provide any explanations or additional information in response. Only answer whatever is asked.**\n
**Try to provide answer in JSON format. For e.g. [{'key_1': 'value_1', 'key_2', 'value_2'}]**
"""

class GoogleGenAIDocIntel():
    def __init__(self, file_path: str):
        self.client = genai.Client(api_key=config.google_genai_api_key)

        # Retrieve and encode the PDF byte
        self.filepath = pathlib.Path(file_path)

    def get_response(self, query):
        response = self.client.models.generate_content(
            model=config.google_genai_model_better,
            contents=[
                types.Part.from_bytes(
                    data=self.filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                query
            ],
            config={
                # 'response_mime_type': 'application/json',
                # 'response_schema': Tariffs,
                'system_instruction': f'{system_prompt}\n{vessel_details}'
            }
        )

        return response
