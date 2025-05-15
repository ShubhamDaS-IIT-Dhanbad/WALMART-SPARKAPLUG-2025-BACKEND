from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from app.core.config import settings
from pydantic import Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dummy embedder that returns a 512-dimensional vector of all 1s
class DummyEmbedder:
    def embed_query(self, text):
        return np.ones(512).tolist()

    def embed_documents(self, texts):
        return [np.ones(512).tolist() for _ in texts]


class PineconeService:
    # _vectorstore = None
    # _embedder = OpenAIEmbeddings(
    #     openai_api_key=settings.OPENAI_API_KEY,
    #     model="text-embedding-3-small",
    #     dimensions=512
    # )
    _vectorstore = None
    _embedder = DummyEmbedder()


    @classmethod
    def get_vectorstore(cls):
        if cls._vectorstore is None:
            try:
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                index = pc.Index(settings.PINECONE_INDEX_NAME)
                cls._vectorstore = index
                logger.info(f"Initialized Pinecone index: {settings.PINECONE_INDEX_NAME}")
            except NotFoundException as e:
                logger.error(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' not found: {str(e)}")
                raise ValueError(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' not found.")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {str(e)}")
                raise ValueError(f"Pinecone initialization failed: {str(e)}")
        return cls._vectorstore

    @classmethod
    def get_retriever(cls, search_kwargs=None):
        index = cls.get_vectorstore()
        k = (search_kwargs or {}).get("k", 4)

        class PineconeRetriever(BaseRetriever):
            index: object = Field(...)
            embedder: OpenAIEmbeddings = Field(...)
            k: int = Field(...)

            def __init__(self, index, embedder, k):
                super().__init__()
                self.index = index
                self.embedder = embedder
                self.k = k

            def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Document retrieval failed: {str(e)}")
                    raise ValueError(f"Document retrieval failed: {str(e)}")

            async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Async document retrieval failed: {str(e)}")
                    raise ValueError(f"Async document retrieval failed: {str(e)}")

        return PineconeRetriever(index=index, embedder=cls._embedder, k=k)





from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from app.core.config import settings
from pydantic import Field
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dummy embedder that returns a 512-dimensional vector of all 1s
class DummyEmbedder:
    def embed_query(self, text):
        return np.ones(512).tolist()

    def embed_documents(self, texts):
        return [np.ones(512).tolist() for _ in texts]

class PineconeService:
    _vectorstore = None
    _embedder = DummyEmbedder()

    @classmethod
    def get_vectorstore(cls):
        if cls._vectorstore is None:
            try:
                # Initialize Pinecone client
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                # Access the index
                index = pc.Index(settings.PINECONE_INDEX_NAME)
                cls._vectorstore = index
                logger.info(f"Initialized Pinecone index: {settings.PINECONE_INDEX_NAME}")
            except NotFoundException as e:
                logger.error(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' not found: {str(e)}")
                raise ValueError(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' not found. Verify the index name or create it.")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {str(e)}")
                raise ValueError(f"Pinecone initialization failed: {str(e)}")
        return cls._vectorstore

    @classmethod
    def get_retriever(cls, search_kwargs=None):
        index = cls.get_vectorstore()
        k = (search_kwargs or {}).get("k", 4)

        class PineconeRetriever(BaseRetriever):
            index: object = Field(...)
            embedder: DummyEmbedder = Field(...)
            k: int = Field(...)

            def __init__(self, index, embedder, k):
                super().__init__(index=index, embedder=embedder, k=k)

            def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    print("1boss")
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Document retrieval failed: {str(e)}")
                    raise ValueError(f"Document retrieval failed: {str(e)}")

            async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                try:
                    vector = self.embedder.embed_query(query)
                    result = self.index.query(vector=vector, top_k=self.k, include_metadata=True)
                    print(result)
                    return [
                        Document(
                            page_content=match["metadata"].get("text", ""),
                            metadata=match["metadata"]
                        )
                        for match in result["matches"]
                    ]
                except Exception as e:
                    logger.error(f"Async document retrieval failed: {str(e)}")
                    raise ValueError(f"Async document retrieval failed: {str(e)}")

        return PineconeRetriever(index, cls._embedder, k)
