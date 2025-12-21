import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
	model="aaditya/Llama2-OpenBioLLM-8B",
	token = HF_TOKEN
)
