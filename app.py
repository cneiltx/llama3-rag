import streamlit as st
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

#url processing
def process_input(urls, question):
    model_local = Ollama(model="llama3")

    url_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in url_list]
    doc_list = [item for sublist in docs for item in sublist]

    # split the documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_split = text_splitter.split_documents(doc_list)

    # convert text chunks into embeddings and store in DB
    vectorstore = Chroma.from_documents(
        documents=doc_split,
        collection_name="rag_chroma",
        embedding=OllamaEmbeddings(model="llama3")
    )
    retriever = vectorstore.as_retriever()

    #perform the rag
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)

#streamlit UI
st.title("Document query with Ollama")
st.write("Enter urls (one per line) and a question to query the documents.")

#input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

#Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)