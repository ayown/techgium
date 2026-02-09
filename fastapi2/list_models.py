
import os
import sys
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: No API key found.")
    sys.exit(1)

print(f"Using API Key: {api_key[:5]}...")

try:
    from google import genai
    print("Imported google.genai")
    
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    
    print("Listing models:")
    try:
        # Pager object, iterate to get models
        for m in client.models.list():
            print(f"- {m.name} (Display: {m.display_name})")
    except Exception as e:
        print(f"Error listing models: {e}")
        # Try v1alpha if v1beta fails? Or just standard v1?
        try:
             client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
             print("Retrying with v1...")
             for m in client.models.list():
                print(f"- {m.name}")
        except Exception as e2:
             print(f"Error listing models with v1: {e2}")

except ImportError as e:
    print(f"Failed to import google.genai: {e}")
    # try google_genai directly?
    try:
        import google_genai
        print("Imported google_genai directly (unexpected but possible)")
    except ImportError:
        print("Could not import google_genai")
