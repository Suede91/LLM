from langchain.document_loaders import DirectoryLoader, WebBaseLoader,UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,TextSplitter
from langchain.vectorstores import Chroma, VectorStore
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings

import itertools

def load_sources() -> list[Document]:
    loaders = {
        '.pdf': UnstructuredPDFLoader(
            "/home/eudes/workspace/pocLLM/toto.pdf"
        ),
    # "sites": WebBaseLoader(
    #     [
    #             "https://news.maxifoot.fr/psg/l-arbitre-contre-milan-est-connu-foot-399907.htm",
    #             # "https://www.ameli.fr/paris/assure/droits-demarches",
    #             # "https://www.cnmss.fr/faq",
    #             # "https://www.cnmss.fr/sites/default/files/2022-10/Livret%20Pratique%20Assur%C3%A9s_octobre2022_1.pdf"
    #         ]
    #     )

        # Add other file types and their respective loaders here
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    all_docs = [loader.load() for loader in loaders.values()]
    return splitter.split_documents(
        list(itertools.chain.from_iterable(all_docs))
    )

def generate_embedding(documents: list[Document], persist_directory: str):
    transformer_model = "all-MiniLM-L6-v2"
    model_embeddings = SentenceTransformerEmbeddings(
        model_name=transformer_model
    )
    return Chroma.from_documents(
        documents,
        model_embeddings,
        persist_directory=persist_directory,
    )


VECTORSTORE_CHROMADB_PATH = 'vectorstore/chromadb'
documents = load_sources()
db = generate_embedding(documents, persist_directory=VECTORSTORE_CHROMADB_PATH)

