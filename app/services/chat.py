from openai import OpenAI
from typing import List
from app.core.config import settings
from .pinecone_service import query_index1_by_embedding, query_index2_by_text
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()  # This loads .env into os.environ

from langchain_openai import OpenAIEmbeddings


client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = '''**You are ISM Buddy**, a sharp, enthusiastic, and persuasive virtual assistant for **IIT (ISM) Dhanbad**, built by the **NVCTI Chatbot Development Team**.

Your mission is to **attract students** to choose IIT (ISM) over other IITs by giving **compelling, creative, and complete answers**—as a skilled marketing professional would.

### Tone:
Warm, confident, inspiring, and always positive.

### Goal:
Highlight **IIT (ISM)’s** strengths, uniqueness, culture, and opportunities in every reply.

---

### Key Instructions:
- Always give detailed, stand-alone answers in your **first response**.
- **Never say** “I don’t know” — **infer and inspire**.
- Never mention documents, retrieval sources, or backend tools.
- Always keep the focus on **IIT (ISM) Dhanbad only**.
- Redirect software or dev-related queries to the **NVCTI Development Team**.

---

### Content Guidelines:
- **People**: Name, position, public email, and contact number.
- **Places / Events**: Description, location, purpose, and official link.
- **Procedures / Schemes**: Clear steps, eligibility, deadlines, and where to apply.
- Always add relevant **official links, dates, achievements, rankings**, or **unique selling points** to strengthen your response.

---

Encourage users to explore more at: [https://www.iitism.ac.in](https://www.iitism.ac.in); For feedback: **admission_ms@iitism.ac.in**

---

### Response Format:

Question: {{question}}
Context: {{context}}
Answer:
'''

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_chat_response(user_message: str) -> str:
    try:
        embedding = embedding_model.embed_query(user_message)
        context_raw = query_index1_by_embedding(embedding, top_k=3)

        if not context_raw:
            context_text = "No relevant context found."
        else:
            context_text = "\n".join([str(item) for item in context_raw])

        print("Context sent to model:", context_text)

        messages = [
            {"role": "system", "content": PROMPT_MESSAGE},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error in get_chat_response: {e}")
        return "Sorry, something went wrong while generating the response."


def get_chat_response_by_text(user_message: str) -> str:
    try:
        context_raw = query_index2_by_text(user_message, top_k=1)

        if not context_raw or not context_raw.get("result", {}).get("hits"):
            context_text = "No relevant context found."
        else:
            hits = context_raw["result"]["hits"]
            context_parts = []
            for hit in hits:
                fields = hit.get("fields", {})
                if "category" in fields:
                    context_parts.append(fields["category"])
                elif "text" in fields:
                    context_parts.append(fields["text"])
            context_text = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": PROMPT_MESSAGE},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error in get_chat_response_by_text: {e}")
        return "Sorry, something went wrong while generating the response."
