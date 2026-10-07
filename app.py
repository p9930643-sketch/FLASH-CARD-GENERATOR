import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env
load_dotenv()

# Get Hugging Face API key
HF_TOKEN = os.getenv("HF_TOKEN")

# Create Inference Client
client = InferenceClient(
    api_key=HF_TOKEN
)

def generate_flashcards(topic, number_of_cards=5):

    prompt = f"""
Create {number_of_cards} flashcards about {topic}.

Return the answer in this format:

Flashcard 1:
Question: ...
Answer: ...

Flashcard 2:
Question: ...
Answer: ...

Create exactly {number_of_cards} flashcards.
Keep the questions simple and the answers accurate.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an educational flashcard generator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message.content


# -------------------------
# USER INPUT
# -------------------------

topic = input("Enter the topic: ")

number = int(
    input("Enter number of flashcards: ")
)

# -------------------------
# BACKEND REQUEST
# -------------------------

flashcards = generate_flashcards(
    topic,
    number
)

# -------------------------
# BACKEND RESPONSE
# -------------------------

print("\n===== GENERATED FLASHCARDS =====\n")

print(flashcards)