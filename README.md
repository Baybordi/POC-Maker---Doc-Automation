# Doc Automation – POC Maker

Chat with your PDFs and automate document workflows using **AWS Bedrock (Claude + Titan)**, **LangChain**, **FAISS**, and **Streamlit**.  
This project extracts knowledge from documents, stores embeddings, and enables interactive Q&A or structured document generation — making it ideal for **document automation** and **Proof of Concept (POC) creation**.

---

## 🚀 Features
- 📄 **PDF ingestion**: Upload and process multiple PDFs.
- 🧩 **Chunking & embeddings**: Split documents into chunks and embed them with **Amazon Titan**.
- 📚 **Vector storage**: Store embeddings locally using **FAISS** for fast retrieval.
- 🤖 **Conversational AI**: Query documents with **Claude (via Bedrock)** using LangChain’s RetrievalQA.
- 🎛 **Streamlit UI**: Simple web interface to manage vector updates and ask questions.
- ⚡ **Automation-ready**: Extendable for structured outputs like meeting minutes, POCs, or summaries.

---

## 🛠️ Tech Stack
- [AWS Bedrock](https://aws.amazon.com/bedrock/) – Claude 3 Sonnet & Titan Embeddings  
- [LangChain](https://www.langchain.com/) – document loaders, embeddings, RetrievalQA  
- [FAISS](https://faiss.ai/) – vector store for similarity search  
- [Streamlit](https://streamlit.io/) – interactive frontend  

---

## 📂 Project Structure




---

## ⚙️ Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/doc-automation-poc-maker.git
   cd doc-automation-poc-maker
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
aws configure
streamlit run app.py
