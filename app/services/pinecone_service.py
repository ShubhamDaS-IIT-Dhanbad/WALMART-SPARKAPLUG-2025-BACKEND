import logging
import numpy as np
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from sentence_transformers import SentenceTransformer
from app.core.config import settings  # Ensure this has your API key and index name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ BERT Embedder using SentenceTransformer
class BertEmbedder:
    def __init__(self, model_name="bert-base-nli-mean-tokens"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()


# ✅ PineconeService with BERT Embedder
class PineconeService:
    _vectorstore = None
    _embedder = BertEmbedder()

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
            embedder: BertEmbedder = Field(...)
            k: int = Field(...)

            def __init__(self, index, embedder, k):
                super().__init__(index=index, embedder=embedder, k=k)

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

        return PineconeRetriever(index, cls._embedder, k)
