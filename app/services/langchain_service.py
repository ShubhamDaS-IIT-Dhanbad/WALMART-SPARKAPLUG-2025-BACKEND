from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.services.pinecone_service import PineconeService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainService:
    _qa_chain = None
    _retriever = None
    _prompt_template = """
You are an intelligent chatbot built for IIT (ISM) Dhanbad.
You help students and visitors by answering questions based on the provided context.

Context:
{context}

Question:
{question}

Answer:"""

    @classmethod
    def get_qa_chain(cls):
        if cls._qa_chain is None:
            try:
                logger.info("Initializing LangChain QA chain...")

                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                )

                if cls._retriever is None:
                    cls._retriever = PineconeService.get_retriever(search_kwargs={"k": 2})

                # Use invoke() instead of deprecated get_relevant_documents()
                test_query = "Sample question to test retriever"
                sample_docs = cls._retriever.invoke({"query": test_query})
                for i, doc in enumerate(sample_docs, 1):
                    logger.info(f"\n--- Sample Retrieved Doc {i} ---\n{doc.page_content}\n")

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=cls._prompt_template
                )

                cls._qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=cls._retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )

                logger.info("Initialized RetrievalQA chain with custom prompt")
            except Exception as e:
                logger.error(f"QA chain initialization failed: {e}")
                raise ValueError(f"QA chain initialization failed: {e}")
        return cls._qa_chain

    @classmethod
    def run_query(cls, query: str):
        try:
            qa_chain = cls.get_qa_chain()

            retriever = cls._retriever or PineconeService.get_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke({"query": query})

            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info("🔍 Retrieved Documents:")
            for i, doc in enumerate(docs, 1):
                logger.info(f"\n--- Document {i} ---\n{doc.page_content}\n")

            final_prompt = cls._prompt_template.replace("{context}", context).replace("{question}", query)
            logger.info(f"\n📨 Final Prompt Sent to LLM:\n{final_prompt}\n")

            result = qa_chain.run(query)
            return result

        except Exception as e:
            context_fallback = context if 'context' in locals() else "N/A"
            error_prompt = cls._prompt_template.replace("{context}", context_fallback).replace("{question}", query)
            logger.error(f"Error during query execution: {e}\nPrompt Used:\n{error_prompt}")

            if "insufficient_quota" in str(e).lower() or "429" in str(e):
                return (
                    "Error: Your OpenAI API quota has been exceeded. "
                    "Please check your plan and billing details at https://platform.openai.com/account/billing/overview\n\n"
                    f"Prompt used:\n{error_prompt}"
                )

            return f"An error occurred while processing your request: {e}\n\nPrompt Used:\n{error_prompt}"


# --------------------
# Fallback example using OpenAI's raw chat completions (if LangChain gives trouble)
# --------------------

def openai_raw_example():
    import openai
    openai.api_key = settings.OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is IIT Dhanbad known for?"}
            ],
            temperature=0.7,
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error occurred: {e}")
