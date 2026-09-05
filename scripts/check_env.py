from dotenv import load_dotenv
import os
from pathlib import Path

# Force load the .env from the current folder
env_path = Path('.') / '.env'
print("Looking for .env at:", env_path.absolute())
print("File exists?", env_path.exists())

load_dotenv(dotenv_path=env_path, override=True)

print("\n--- Loaded values ---")
print("TOMTOM_API_KEY     →", repr(os.getenv("TOMTOM_API_KEY")))
print("OPENWEATHER_API_KEY →", repr(os.getenv("OPENWEATHER_API_KEY")))
print("SUPABASE_URL       →", repr(os.getenv("SUPABASE_URL")))
print("SUPABASE_KEY       →", repr(os.getenv("SUPABASE_KEY")[:20] + "..." if os.getenv("SUPABASE_KEY") else None))
print("GROQ_API_KEY       →", repr(os.getenv("GROQ_API_KEY")[:20] + "..." if os.getenv("GROQ_API_KEY") else None))