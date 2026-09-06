from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client (ensure GROQ_API_KEY environment variable is set)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing.")
        
client = Groq(api_key=api_key)

# 1. Ingestion Phase: Define custom source documents locally
knowledge_base = [
    "Project Titan's Q3 budget allocation is $450,000, managed by Sarah.",
    "Project Phoenix focuses on migrating legacy infrastructure to AWS by November.",
    "Company remote policy allows working from anywhere within the home country for up to 60 days a year."
]

# 2. Retrieval Phase: Simple keyword/relevance match retriever
def retrieve_context(query, documents):
    query_words = set(query.lower().split())
    best_doc = max(documents, key=lambda doc: len(query_words.intersection(doc.lower().split())))
    return best_doc

user_query = "Who is Sagar Gavand?"
retrieved_context = retrieve_context(user_query, knowledge_base)

# 3. Generation Phase: Send context and query to Groq's high-speed LLM
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question accurately using ONLY the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{retrieved_context}\n\nQuestion: {user_query}"
        }
    ],
    model="openai/gpt-oss-20b",
)

print(f"Query: {user_query}")
print(f"Retrieved Context: {retrieved_context}")
print(f"Groq Answer: {chat_completion.choices[0].message.content}")