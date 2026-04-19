# Aria — HR Policy Assistant for NovaTech Solutions

An AI-powered HR chatbot built with LangGraph, ChromaDB, and Groq (LLaMA 3.3 70B).

## Features
- RAG-based policy answering (12 HR policy documents)
- Leave Balance Calculator tool (live date calculations)
- Conversation memory with multi-turn support
- Self-reflection eval node for faithfulness scoring
- Streamlit chat UI

## Tech Stack
LangGraph | ChromaDB | Groq | SentenceTransformer | RAGAS | Streamlit

## Setup
1. Clone the repo
2. Create a `.env` file with your Groq API key:
3. 
GROQ_API_KEY=your_key_here


4. Install dependencies:
pip install langgraph langchain-groq chromadb sentence-transformers ragas streamlit python-dotenv langchain-huggingface datasets


5. Run the app:
streamlit run capstone_streamlit.py


## Evaluation
| Metric | Score |
|---|---|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.807 |
| Context Precision | 1.000 |
