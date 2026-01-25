import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient

# Constants
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Sample documentation from the original app
SAMPLE_DOCS = [
    {"title": "Turbine Maintenance Manual", "content": "Daily Inspections: Check oil levels (75-85%), verify blade pitch (±0.5°), monitor vibration (<2.5 mm/s). Weekly: Lubricate bearing, inspect hydraulic lines. Monthly: Oil analysis, blade crack inspection."},
    {"title": "Safety Procedures", "content": "Confined Space Entry: Complete Entry Permit (CSE-001), verify oxygen (19.5-23.5%), LEL (<10%). Required PPE: Hard hat, safety harness, retrieval line, communication device."},
    {"title": "Equipment Specs", "content": "GE Frame 7FA Gas Turbine Compressor: 171 MW power, 9,790 BTU/kWh heat rate, 15.5:1 pressure ratio. Startup time: 20 mins. Overhaul interval: 48,000 hours."}
]

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.texts = [d["content"] for d in SAMPLE_DOCS]
        self.metadatas = [{"source": d["title"]} for d in SAMPLE_DOCS]
        
        # In-memory vector store for the demo
        self.vectorstore = Chroma.from_texts(
            texts=self.texts,
            embedding=self.embeddings,
            metadatas=self.metadatas
        )
        
        self.client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

    def query(self, message: str) -> str:
        # 1. Similarity search
        docs = self.vectorstore.similarity_search(message, k=2)
        context = "\n\n".join([f"SOURCE: {d.metadata['source']}\n{d.page_content}" for d in docs])
        
        # 2. Augmented Generation
        prompt = f"""You are a Senior Industrial Systems Specialist. Answer the user's question using the provided technical context.
        If the information is not in the context, use your general knowledge but specify it's based on general industry standards.

CONTEXT:
{context}

USER QUESTION:
{message}

SENIOR EXPERT ANSWER:"""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        return response.choices[0].message.content

rag_engine = RAGEngine()
