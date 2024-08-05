# Import Modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from ragas_eval import get_test_set, do_evaluate

import os
import warnings

os.environ["MISTRAL_API_KEY"] = "ZlWLv7v514TvKDC6kWnvr2kBfD6eM02B"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
warnings.filterwarnings(action="ignore")


# Method that builds DB based on documents and returns database's retriever
def get_retriever(doc_path, embedding_model):

    # Load data source (e.g. PDF)
    pdfLoader = PyPDFLoader(doc_path)
    raw_data = pdfLoader.load()

    # Make raw_data into chunk splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_data)

    # Define language embedding model
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

    # Create a Database to retrieve embedding vectors efficiently
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Create an instance of retriever
    chroma_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
    )

    return vectorstore, chroma_retriever


# Method for prompt engineering
def get_prompts():

    # 1. Default prompt for question-answering task
    template = """You are an assistant for question-answering tasks.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Answer:
    """

    # 2. Prompt that utilizes context to generate an appropriate answer
    context_template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """

    # 3
    augmented_prompt = """Using the contexts below, answer the query.
    query: {query}
    contexts: {source_knowledge}
    """

    prompt = ChatPromptTemplate.from_template(template)
    context_prompt = ChatPromptTemplate.from_template(context_template)
    source_prompt = PromptTemplate(
        template=augmented_prompt, input_variables=["query", "source_knowledge"]
    )
    return prompt, context_prompt, source_prompt


# Build langchain for question-answering task
def build_langchain(llm_id, prompt, context_prompt, augmented_prompt, retriever):

    # Use API model for langchain's text generation endpoint
    llm = ChatMistralAI(model=llm_id)
    llm_hub = HuggingFaceHub(repo_id="google/flan-t5-base")

    # Generate 2 kinds of chain
    # 1. General llm chain
    chain = prompt | llm | StrOutputParser()

    # 2. RAG chain
    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | context_prompt
        | llm
        | StrOutputParser()
    )

    # HuggingFace chain
    qa_chain = LLMChain(
        prompt=augmented_prompt,
        llm=llm_hub,
        llm_kwargs={"temperature": 0, "max_length": 1024},
    )

    return chain, rag_chain, qa_chain


# Main flow of langchain-RAG for question-answering
if __name__ == "__main__":

    # Local field
    test_path = "../datasets/benchmark.json"
    pdf_path = "../datasets/bert.pdf"
    embedding_model = "all-MiniLM-L6-v2"
    llm_id = "mistral-large-latest"

    # Given data source & embedding model, return the DB's retriever
    vector_store, db_retriever = get_retriever(pdf_path, embedding_model)

    # Get prompts to proceed prompt engineering
    prompt, context_prompt, augmented_prompt = get_prompts()

    # Build 2 kinds of chain for question-answering task
    chain, rag_chain, qa_chain = build_langchain(
        llm_id, prompt, context_prompt, augmented_prompt, db_retriever
    )

    # query = "What are the two main tasks BERT is pre-trained on?"
    # print(rag_chain.invoke(query))
    # print(rag_chain.invoke(query))
    # result_simi = db_retriever.get_relevant_documents(query)
    # result_simi = vector_store.similarity_search(query, k=1)
    # source_knowledge = "\n".join([x.page_content for x in result_simi])
    # answer = qa_chain.run({"source_knowledge": source_knowledge, "query": query})
    # print(source_knowledge)
    # print("------------------------------------------------------------")
    # print(answer)

    # evaluates langchain-RAG with 3 main text generation metrics
    questions, ground_truths = get_test_set(test_path)
    total_Precision, total_Recall, total_F1 = do_evaluate(
        rag_chain, db_retriever, questions, ground_truths
    )

    print(f"Total Precision: {total_Precision:.4f}")
    print(f"Total Recall: {total_Recall:.4f}")
    print(f"Total F1 Score: {total_F1:.4f}")
