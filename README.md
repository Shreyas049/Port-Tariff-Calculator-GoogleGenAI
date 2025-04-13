This is GenAI application to calculate the port tariff information for the vessel birthing at a certain port. Application uses agentic approach where 1st agent extracts text from "Port Tariff.pdf" document provided by authrorities in required format. And 2nd agent uses RAG approach to find out the relevent information about the vessel given user's query.

How to run:
I. Add proper configurations in ./conf/config.py file. Currently "GOOGLE_GENAI_API_KEY" is needed.
II. Make sure you have the tariff document in ./data directory.
III. Adjust the user_prompt according to your need in ./prompts.py file.
IV. Make sure all required paths are correct in main.py file
V. Install all dependencies in requirements.txt file
VI. Run main.py file using "python main.py" command. Output should be printed on console. 



Directory Structure:

Main Repo :
    - conf : DIR
        -- config.py : Contains LLM Model and API KEY. Make changes according to requirements.
    - data : DIR
        -- Contains "Port Tariff.pdf" & "Port Tariff.txt"
    - model : DIR
        -- agent_document_loader.py : Document Loader agent that extracts text from pdf file and saves as txt.
        -- agent_porttariff_engine.py : PortTariffEngine class where RAG model is defined.
    - prompts.py : file : Contains user_prompt
    - main.py : file : Run main.py to execute the pipeline.


# Port Tariff Calculator

## Overview

The Port Tariff Calculator is a GenAI application designed to calculate port tariff information for vessels birthing at specific ports. The application utilizes an agentic approach with two specialized agents:

1. **Document Loader Agent**: Extracts text from "Port Tariff.pdf" documents provided by port authorities and converts them to a structured format.
2. **Port Tariff RAG Agent**: Uses a Retrieval-Augmented Generation (RAG) approach to find relevant tariff information based on vessel details provided in the user's query.

## How to Run

### Prerequisites

1. Google Gemini API key
2. Python 3.10+
3. Port Tariff documents in PDF format

### Setup

1. **Configure API Key**
   - Add proper configurations in `./conf/config.py` file
   - Currently, `GOOGLE_GENAI_API_KEY` is required

2. **Prepare Documents**
   - Make sure you have the tariff document in the `./data` directory
   - The system expects a file named "Port Tariff.pdf"

3. **Adjust Prompts**
   - Customize the user_prompt in `./prompts.py` according to your needs

4. **Check Paths**
   - Verify all required paths are correct in `main.py`

5. **Install Dependencies**
   - Install all dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Application**
   - Execute the application using:
   ```bash
   python main.py
   ```
   - The output will be printed to the console

## Directory Structure

```
Main Repo
├── conf/
│   └── config.py                # Contains LLM Model and API KEY configurations
├── data/
│   ├── Port Tariff.pdf          # Source PDF document from port authorities
│   └── Port Tariff.txt          # Extracted text from PDF (generated)
├── model/
│   ├── agent_document_loader.py # Document Loader agent that extracts text from PDF
│   └── agent_porttariff_engine.py # PortTariffEngine class with RAG implementation
├── prompts.py                   # Contains user query prompts
├── main.py                      # Main execution script
└── requirements.txt             # Dependencies
```


## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
