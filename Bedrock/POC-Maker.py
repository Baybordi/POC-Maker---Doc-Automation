import os
import boto3
import streamlit as st

from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock client with region and credentials
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-2")

# Embedding model ID (Titan text embedding v2)
MODEL_ID_EMBEDDING = "amazon.titan-embed-text-v2:0"

# Chat LLM model ID (Claude v3 example)
MODEL_ID_LLM = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize embeddings
bedrock_embedding = BedrockEmbeddings(model_id=MODEL_ID_EMBEDDING, client=bedrock_client)

def data_ingestion():
    if not os.path.exists("data"):
        st.error("Data folder not found! Please create a 'data' folder and add PDFs.")
        return None
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    if docs is None:
        return None
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embedding)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_claude_llm():
    try:
        llm = BedrockChat(
            model_id=MODEL_ID_LLM,
            client=bedrock_client,
            model_kwargs={"max_tokens": 512}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Bedrock Chat LLM: {str(e)}")
        return None

# Prompt template for RetrievalQA chain
custom_prompt = PromptTemplate(
    template="""Use the context below to answer the question:

Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    if llm is None or vectorstore_faiss is None:
        return "Model or vector store not ready."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="Chat PDF with AWS Bedrock")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question from the PDF file")

    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                if docs:
                    get_vector_store(docs)
                    st.success("Vector store updated!")

    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            if not user_question.strip():
                st.warning("Please enter a question!")
            else:
                try:
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embedding, allow_dangerous_deserialization=True)
                    llm = get_claude_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.markdown("### Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()
