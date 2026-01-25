import gradio as gr
import os
from src.rag_engine import rag_engine

def chat_response(message, history):
    try:
        response = rag_engine.query(message)
        return response
    except Exception as e:
        return f"### Error: {str(e)}\n\nPlease ensure your HF_TOKEN is correctly configured."

# ============================================
# GRADIO UI
# ============================================

with gr.Blocks(title="Chat with Assets RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Chat with Assets RAG
    ### Enterprise Knowledge Retrieval
    
    This agent uses **ChromaDB** for vector retrieval and **Mistral-7B** for expert-level responses based on technical manuals.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                examples=[
                    "What is the daily maintenance checklist for the turbine?",
                    "What PPE is required for confined space entry?",
                    "What are the specifications for the GE compressor?",
                    "When should we do a major overhaul?"
                ],
                title=""
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Knowledge Base")
            gr.Markdown("""
            **Connected Assets:**
            
            1. **Turbine Maintenance Manual**
               - Real-time retrieval of limit values and schedules.
            
            2. **Safety Procedures (CSE)**
               - Compliance verification for hazardous zones.
            
            3. **Equipment Specifications**
               - Deep technical data on GE Frame 7FA units.
            """)
    
    gr.Markdown("""
    ---
    **Tech Stack:** LangChain • ChromaDB • Mistral-7B • Gradio
    
    **Author:** [David Fernandez](https://davidfernandez.dev) | AI Engineer
    """)

if __name__ == "__main__":
    demo.launch()
