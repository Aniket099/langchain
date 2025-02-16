#!/usr/bin/env python
# coding: utf-8

# Install Required Libraries
import subprocess
import sys

# Install necessary dependencies
required_packages = ["boto3", "sagemaker", "langchain", "streamlit"]

for package in required_packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", package], check=True)

# Import Necessary Libraries
import boto3
import sagemaker
import json
import fitz  # PyMuPDF for PDF processing
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler
import tiktoken

# Set Up AWS SageMaker Session and Role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Deploy Hugging Face Model on SageMaker
from sagemaker.huggingface import HuggingFaceModel

# Define Hugging Face model configuration
hub = {
    'HF_MODEL_ID': 'facebook/bart-large-cnn',  # Summarization model
    'HF_TASK': 'summarization'
}

# Use newer Python & PyTorch versions
huggingface_model = HuggingFaceModel(
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py38",
    env=hub,
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy model to an endpoint
predictor = huggingface_model.deploy(initial_instance_count=1, instance_type="ml.m5.large")
endpoint_name = predictor.endpoint_name
print(f"Model deployed at: {endpoint_name}")

# Define LangChain Content Handler
class CustomContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        request_dict = {"inputs": prompt, **model_kwargs}
        return json.dumps(request_dict).encode("utf-8")

    def transform_output(self, response: bytes) -> str:
        response_dict = json.loads(response.decode("utf-8"))
        return response_dict[0]["summary_text"]  # Extract summarized text

# Initialize LangChain model
content_handler = CustomContentHandler()
llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    region_name="eu-west-1",
    model_kwargs={"temperature": 0.7, "max_length": 200},
    content_handler=content_handler
)

print("LangChain model connected to SageMaker endpoint")

# Extract & Chunk Large PDFs
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF and returns it as a string."""
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    return text.strip()

def chunk_text(text, max_tokens=500):
    """Splits text into smaller chunks based on token limits."""
    tokenizer = tiktoken.get_encoding("cl100k_base")  # For token estimation
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(tokenizer.encode(" ".join(chunk))) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

# Optimized Summarization for Small & Large PDFs
def summarize_large_text(text, max_tokens=500):
    """Summarizes text efficiently by checking its size."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_count = len(tokenizer.encode(text))

    if token_count <= max_tokens:
        # Directly summarize if within model limits
        return llm.invoke(text)
    
    # Otherwise, use chunking for large PDFs
    chunks = chunk_text(text, max_tokens)
    summaries = [llm.invoke(chunk) for chunk in chunks]

    # Merge summarized chunks into a final summary
    final_summary = llm.invoke(" ".join(summaries))
    return final_summary
