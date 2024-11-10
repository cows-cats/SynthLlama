from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
import tempfile
import os
import chromadb

from dotenv import load_dotenv


load_dotenv()

# Security improvement: Validate environment variable
if not (together_api_key := os.getenv("TOGETHER_API_KEY")):
    raise EnvironmentError("TOGETHER_API_KEY environment variable not set.")

app = FastAPI()

embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key=together_api_key,
)
llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    api_key=together_api_key,
)

# Use try-except block to handle database initialization errors
try:
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("embeddings")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")


class DataRequest(BaseModel):
    format_type: str
    data_amount: int
    description: Optional[str] = None

def get_format_prompt(format_type: str, data_amount: int) -> str:
    base_prompts = {
        "qa_format": "Generate {amount} question-answer pairs in the following format:\n{{\"question\": \"...\", \"answer\": \"...\"}}\nMake sure the data is relevant to: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "chat_format": "Generate a chat conversation with {amount} messages in the following format:\n[{{\"role\": \"user\", \"content\": \"...\"}}, {{\"role\": \"assistant\", \"content\": \"...\"}}]\nMake the conversation relevant to: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "completion_format": "Generate {amount} prompt-completion pairs in the following format:\n{{\"prompt\": \"...\", \"completion\": \"...\"}}\nBased on the context: {context}, only answer with the format and make sure it is compatible with json , do not add any other text",
        "text_classification_format": "Generate {amount} classified text examples in the following format:\n{{\"text\": \"...\", \"label\": \"...\"}}\nMake the examples relevant to: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "translation_format": "Generate {amount} translation pairs in the following format:\n{{\"source\": \"...\", \"target\": \"...\"}}\nConsider the context: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "instruction_response_format": "Generate {amount} instruction-response pairs in the following format:\n{{\"instruction\": \"...\", \"input\": \"...\", \"output\": \"...\"}}\nBased on: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "summarization_format": "Generate {amount} article-summary pairs in the following format:\n{{\"article\": \"...\", \"summary\": \"...\"}}\nRelated to: {context}, only answer with the format and make sure it is compatible with json, do not add any other text",
        "dialogue_format": "Generate a dialogue with {amount} exchanges in the following format:\n[{{\"speaker\": \"A\", \"text\": \"...\"}}, {{\"speaker\": \"B\", \"text\": \"...\"}}]\nMake it relevant to: {context}, only answer with the format and make sure it is compatible with json, do not add any other text"
    }
    return base_prompts.get(format_type, "Invalid format type")

@app.post("/process")
async def process_request(request: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        # Parse the request data
        request_data = json.loads(request)
        request_obj = DataRequest(**request_data)

        # Validate format type early
        prompt_template = get_format_prompt(request_obj.format_type, request_obj.data_amount)
        if prompt_template == "Invalid format type":
            raise HTTPException(status_code=400, detail="Invalid format type specified.")

        # Handle PDF if provided
        context = ""
        query_engine = None
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await file.read()
                temp_file.write(content)

            uploaded_dir = "uploaded_files"

            if not os.path.exists(uploaded_dir):
                os.makedirs(uploaded_dir)

            temp_file_name = temp_file.name

            temp_file_path = os.path.join(uploaded_dir, temp_file_name)


            try:
                parser = LlamaParse(
                result_type="markdown"  # "markdown" and "text" are available
                )
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()

                #documents = LlamaParse.load_data(file_path=temp_file_path)

                index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context,
                    embed_model=embed_model
                )
                query_engine = index.as_query_engine()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
        else:
            context = request_obj.description if request_obj.description else ""

        # Format the prompt
        prompt = prompt_template.format(
            amount=request_obj.data_amount,
            context=context
        )

        # Generate data
        try:
            if query_engine:
                response = query_engine.query(prompt)
            else:
                response = llm.complete(prompt)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during query generation: {str(e)}")

        # Validate JSON response
        try:
          # Validate JSON
            return {"status": "success", "data": response.text}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Generated data is not valid JSON.")

    except HTTPException as e:
        # Re-raise HTTP exceptions for correct status codes
        raise e
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/formats")
async def get_formats():
    formats = [
        "qa_format",
        "chat_format",
        "completion_format",
        "text_classification_format",
        "translation_format",
        "instruction_response_format",
        "summarization_format",
        "dialogue_format"
    ]
    return {"formats": formats}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
