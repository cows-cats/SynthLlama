from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import os
import together
from dotenv import load_dotenv

load_dotenv()

together.api_key = os.getenv("TOGETHER_API_KEY")

app = FastAPI()

class DataRequest(BaseModel):
    format_type: str
    data_amount: int
    description: Optional[str] = None

def get_format_prompt(format_type: str, data_amount: int) -> str:
    """Generate appropriate prompt based on format type"""
    base_prompts = {
        "qa_format": """Generate {amount} question-answer pairs in the following format:
            {{"question": "...", "answer": "..."}}
            Make sure the data is relevant to: {context}""",
            
        "chat_format": """Generate a chat conversation with {amount} messages in the following format:
            [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]
            Make the conversation relevant to: {context}""",
            
        "completion_format": """Generate {amount} prompt-completion pairs in the following format:
            {{"prompt": "...", "completion": "..."}}
            Based on the context: {context}""",
            
        "text_classification_format": """Generate {amount} classified text examples in the following format:
            {{"text": "...", "label": "..."}}
            Make the examples relevant to: {context}""",
            
        "translation_format": """Generate {amount} translation pairs in the following format:
            {{"source": "...", "target": "..."}}
            Consider the context: {context}""",
            
        "instruction_response_format": """Generate {amount} instruction-response pairs in the following format:
            {{"instruction": "...", "input": "...", "output": "..."}}
            Based on: {context}""",
            
        "summarization_format": """Generate {amount} article-summary pairs in the following format:
            {{"article": "...", "summary": "..."}}
            Related to: {context}""",
            
        "dialogue_format": """Generate a dialogue with {amount} exchanges in the following format:
            [{{"speaker": "A", "text": "..."}}, {{"speaker": "B", "text": "..."}}]
            Make it relevant to: {context}"""
    }
    
    return base_prompts.get(format_type, "Invalid format type")


def process_pdf(file_path: str) -> List:
    """Process PDF and create document chunks"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def create_faiss_index(documents: List):
    """Create FAISS index from documents"""
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db


def query_llama(prompt: str, context: str) -> str:
    """Query LLaMA model using Together AI"""
    full_prompt = f"""Context: {context}

Task: {prompt}

Please generate the data in valid JSON format.

Response:"""
    
    response = together.Complete.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        prompt=full_prompt,
        max_tokens=2048,  # Increased for larger responses
        temperature=0.7,
    )
    
    return response['output']['choices'][0]['text']


@app.post("/process")
async def process_request(request: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        # Parse the request data
        request_data = json.loads(request)
        request_obj = DataRequest(**request_data)
        
        # Handle PDF if provided
        context = ""
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                documents = process_pdf(temp_file_path)
                db = create_faiss_index(documents)
                
                if request_obj.description:
                    relevant_docs = db.similarity_search(request_obj.description, k=3)
                    context = " ".join([doc.page_content for doc in relevant_docs])
                else:
                    context = " ".join([doc.page_content for doc in documents[:3]])
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
        else:
            context = request_obj.description if request_obj.description else ""

        # Get the appropriate prompt template
        prompt_template = get_format_prompt(request_obj.format_type, request_obj.data_amount)
        if prompt_template == "Invalid format type":
            return {"status": "error", "message": "Invalid format type specified"}
            
        # Format the prompt
        prompt = prompt_template.format(
            amount=request_obj.data_amount,
            context=context
        )
        
        # Generate data
        response = query_llama(prompt, context)
        
        # Validate JSON response
        try:
            json.loads(response)  # Validate JSON
            return {"status": "success", "data": response}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Generated data is not valid JSON"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/formats")
async def get_formats():
    """Return available format types"""
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