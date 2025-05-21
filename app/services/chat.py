from openai import OpenAI
from app.core.config import settings
from .pinecone_service import query_pinecone
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Safe download of NLTK resources (only if not already downloaded)
def ensure_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()

client = OpenAI(api_key=settings.OPENAI_API_KEY)

PROMPT_MESSAGE = (
    "You are a chatbot for IIT ISM Dhanbad. Only answer questions related to the institute scholrship,hostel,academics. "
    "If the question is unrelated, reply: 'I can only answer questions about IIT ISM Dhanbad.' Be brief and accurate."
)

# NLTK Preprocessing
def preprocess(text: str) -> str:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered = [
        stemmer.stem(w)
        for w in tokens
        if w not in stop_words and w not in string.punctuation
    ]
    return " ".join(filtered)

def get_embedding(text: str) -> list[float]:
    processed_text = preprocess(text)
    response = client.embeddings.create(model="text-embedding-ada-002", input=processed_text)
    return response.data[0].embedding

def get_chat_response(user_message: str) -> str:
    embedding = get_embedding(user_message)
    context_raw = query_pinecone(embedding, top_k=3)
    
    # Safely extract text from Pinecone response
    context_text = context_raw.get("text", "") if isinstance(context_raw, dict) else context_raw

    context = preprocess(context_text)

    messages = [
        {"role": "system", "content": PROMPT_MESSAGE},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_message}"}
    ]

    # 🔹 Print the ultimate prompt
    print("\n=== Final Prompt Sent to OpenAI ===",context,messages)

    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content
