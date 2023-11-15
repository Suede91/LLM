import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


import chromadb
from camelot.core import TableList, Table
from langchain.document_loaders import WebBaseLoader, DataFrameLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from llmsherpa.readers import LayoutPDFReader
from pandas import DataFrame



class DocType(Enum):
    PDF = 0,
    WEB = 1,
    TEXT = 2,
    DIR = 3

model = "all-MiniLM-L12-v2"
# model = "sentence-t5-xxl"
model2 = "multi-qa-mpnet-base-dot-v1"
facebook_model = 'facebook/bart-large'
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
resources = {
    DocType.PDF: [
        "/home/eudes/workspace/pocLLM/toto.pdf"
        #   "resources/data/doc/UNEO-REFERENCE/RM_Unéo_Référence_Unéo_International_juillet 23.pdf",
        #   "resources/data/doc/UNEO-REFERENCE/RM_U-R-U-i_janvier2023_2211.pdf",
        #   "resources/data/doc/UNEO-REFERENCE/RM_U-C_janvier2023_2211.pdf"
    ]
}

VECTORSTORE_CHROMADB_PATH = 'vectorstore/chromadb'

class ChromaEmbedding:
    default_config = {
        "search_type": "mmr",  # "similarity", "similarity_score_threshold".
        "search_kwargs": {"k": 5},
        # k: Amount of documents to return (Default: 4)
        # score_threshold: Minimum relevance threshold for similarity_score_threshold
        # fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
        # lambda_mult: Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)
        # filter: Filter by document metadata
    }

    def __init__(self, model_name: str,
                 documents: Dict[DocType, list[str]],
                 chunk_size=4000,
                 chunk_overlap=100,
                 client: Optional[chromadb.Client] = None
                 ) -> None:
        self.transformer = SentenceTransformerEmbeddings(model_name=model_name)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        self.documents = ChromaEmbedding._loadDocuments(documents, splitter)
        self.embedding = Chroma.from_documents(
            documents=self.documents,
            client=client,
            embedding=self.transformer)

    def retriever(self, config=None):
        _config = self.default_config if config is None else config
        return self.embedding.as_retriever(**_config)

    def getCollection(self, collection_name):
        return self.embedding.client.get_or_create_collection(collection_name)

    @staticmethod
    def _loadDocuments(documents: Dict[DocType, list[str]], splitter: TextSplitter):
        import itertools
        loading_doc: list[Document] = []
        for type_doc, values in documents.items():
            match type_doc:
                case DocType.PDF:
                    pdf = [ChromaEmbedding.load_pdf_using_layout_lm(str(f)) for f in values]
                    docs = list(itertools.chain.from_iterable(pdf))
                case DocType.WEB:
                    docs = WebBaseLoader(values).load()
                case DocType.DIR:
                    all_pdf = Path(values[0]).rglob("**/*.pdf")
                    chuncks = []
                    for pdf in all_pdf:
                        chuncks.append(ChromaEmbedding.load_pdf_using_layout_lm(str(pdf)))
                    docs = list(itertools.chain.from_iterable(chuncks))
                case _:
                    docs: list[Document] = []

            loading_doc.extend(docs)
        loading_doc = sorted(loading_doc, key=lambda x: len(x.page_content), reverse=True)
        return loading_doc

    @staticmethod
    def load_pdf_using_layout_lm(file: str):
        print(f"Load file test {file}")
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(file)
        return [Document(page_content=chunk.to_text(True,True)) for chunk in
                doc.sections()]

    def cleanDocument(self, document: Document):
        import regex as re
        content = document.page_content
        document.page_content = re.sub(r'\n{3, 10}', '\n', content)

client = chromadb.PersistentClient(VECTORSTORE_CHROMADB_PATH)
store = ChromaEmbedding(model_name=model2,documents=resources,client=client)
print("End embedding")
