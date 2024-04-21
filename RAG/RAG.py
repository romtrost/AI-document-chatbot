import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

CWD = os.getenv("CWD")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

class RAG:
    def __init__(
        self,
        filename = "",
        llm_model = "gpt-3.5-turbo-0125",
        embedding_model = "text-embedding-3-large",
        chunk_size = 2500,
        chunk_overlap = 250,
    ):
        self.document_path = CWD + "Documents/" + filename
        self.database_path = CWD + "RAG/vectordb/" + filename
        self.llm = ChatOpenAI(model=llm_model,
                              openai_api_key=OPENAI_API_KEY)
        self.embedding = OpenAIEmbeddings(model=embedding_model,
                                          openai_api_key=OPENAI_API_KEY)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self, query):
        """ Runs the RAG main loop. """

        print("--- Starting RAG... ---")

        # load document
        loader = PyPDFLoader(self.document_path)
        documents = loader.load()
        # chunk document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        # create or load vector DB
        if os.path.exists(self.database_path):
            print("Loading vector database...")
            vectordb = Chroma(persist_directory=self.database_path,
                              embedding_function=self.embedding)
        else:
            print("Creating vector database...")
            vectordb = Chroma.from_documents(documents=chunks,
                                             embedding=self.embedding,
                                             persist_directory=self.database_path)
        # set up vector DB as retriever
        retriever = vectordb.as_retriever(search_type="similarity")  
        # create QA chain
        qa = RetrievalQA.from_chain_type(llm=self.llm,
                                         chain_type="stuff",
                                         retriever=retriever,
                                         return_source_documents=True)
        # trigger QA chain using query
        response = qa({"query": query})
        # retrieve result and sources
        result = response["result"]
        sources = response["source_documents"]

        return result, sources
    