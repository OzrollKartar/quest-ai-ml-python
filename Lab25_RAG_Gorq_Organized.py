import os
import logging
import pyodbc
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_sql_connection():
    try:
        # Note: Replace 'YourPasswordHere' with your actual 'sa' account password
        connection_string = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=DESKTOP-S2S4KJ2;"
            "Database=EfDemoDb;"  # Specify your target database name
            "UID=sa;"
            "PWD=aa;"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )
        conn = pyodbc.connect(connection_string)
        return conn
    except Exception as e:
        logger.error(f"SQL Server connection failed: {e}")
        raise
def retrieve_from_sql_server(query, top_k=3):
    conn = get_sql_connection()
    cursor = conn.cursor()
    
    try:
        # Filter out stop words and keep meaningful search terms
        stop_words = {"what", "is", "the", "for", "a", "an", "in", "on", "of"}
        keywords = [word.strip("?.,!") for word in query.split() if word.lower() not in stop_words and len(word) > 2]
        
        if not keywords:
            keywords = [query]
            
        # Build dynamic SQL search matching any of the key terms
        sql_query = """
            SELECT TOP (?) Content 
            FROM KnowledgeBase 
            WHERE Content LIKE ? OR Content LIKE ?
        """
        
        # Search for primary keywords (e.g., Project and Titan)
        param1 = f"%{keywords[-1]}%"  # Usually the subject like 'Titan'
        param2 = f"%{keywords[0]}%"   # Context word
        
        cursor.execute(sql_query, (top_k, param1, param2))
        rows = cursor.fetchall()
        
        docs = [row.Content for row in rows]
        
        # Fixed TypeError by using f-string formatting
        print(f"Total docs found: {len(docs)}")
        
        if not docs:
            return "No matching records found in SQL Server database."
        return " ".join(docs)
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return "Error fetching database context."
    finally:
        cursor.close()
        conn.close()

def generate_rag_response(query):
    retrieved_context = retrieve_from_sql_server(query)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
        
    client = Groq(api_key=api_key)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an enterprise database assistant. Answer accurately using ONLY the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{retrieved_context}\n\nQuestion: {query}"
                }
            ],
            model="openai/gpt-oss-20b",
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "Service temporarily unavailable."

if __name__ == "__main__":
    user_query = "What is the budget for Project Titan?"
    answer = generate_rag_response(user_query)
    print(f"\nQuery: {user_query}")
    print(f"Answer: {answer}")