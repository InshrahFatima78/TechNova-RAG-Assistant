# 🚀 TechNova RAG AI Assistant

An intelligent Retrieval-Augmented Generation (RAG) assistant built to answer questions using TechNova's dedicated business knowledge base.

## ✨ Features

- 🤖 AI-powered conversational assistant
- 📚 Retrieval-Augmented Generation (RAG)
- 🔎 Knowledge-base document retrieval
- 💬 Natural-language question answering
- 💰 Pricing and service information
- 🏢 Company information
- ❓ FAQ and support information
- 📋 Refund and support policy assistance
- 🔐 Secure API-key configuration
- 🌐 Streamlit web interface

## 🧠 How It Works

The application follows a RAG pipeline:

User Question  
↓  
Document Retrieval  
↓  
Relevant Context  
↓  
LLM Processing  
↓  
Grounded AI Response

The assistant retrieves relevant information from the TechNova knowledge base before generating an answer, helping keep responses relevant to the available business information.

## 📚 Knowledge Base

The RAG system uses the following TechNova documents:

- `company_overview.txt`
- `pricing.txt`
- `services.txt`
- `faq.txt`
- `refund_support_policy.txt`

All knowledge-base files are stored inside the `documents/` directory.

## 🛠️ Technology Stack

- **Python**
- **Streamlit** — web application interface
- **LangChain** — RAG application framework
- **ChromaDB** — vector database
- **Sentence Transformers** — text embeddings
- **Groq** — language model API
- **python-dotenv** — environment configuration

## 📁 Project Structure

```text
TechNova-RAG-Assistant/
│
├── documents/
│   ├── company_overview.txt
│   ├── pricing.txt
│   ├── services.txt
│   ├── faq.txt
│   └── refund_support_policy.txt
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
