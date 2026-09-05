from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_groq import ChatGroq
from groq import APIStatusError

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before running this test.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
)

try:
    response = llm.invoke("Say hello from Masar in one long sentence.")
except APIStatusError as error:
    raise RuntimeError(f"Groq request failed: {error}") from error

print(response.content)