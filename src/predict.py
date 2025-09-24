import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process

# Load your trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv("../data/symptoms_diseases.csv")

# Optional: FAQ dictionary
faq_dict = {
    "what is flu": "Flu is a viral infection affecting the respiratory system.",
    "how to prevent cold": "Wash hands, avoid close contact with sick people, and stay hydrated."
}

print("Healthcare Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("Enter symptoms (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        break

    # Check FAQ first
    if user_input.lower() in faq_dict:
        print(f"Answer: {faq_dict[user_input.lower()]}\n")
        continue

    # Find top 3 closest symptom matches using rapidfuzz
    matches = process.extract(user_input, df['symptoms'].tolist(), limit=3, score_cutoff=50)

    if not matches:
        print("No matching symptoms found. Please try different words.\n")
        continue

    print("Possible diseases:")
    for match in matches:
        symptom_text = match[0]
        disease = df[df['symptoms'] == symptom_text]['disease'].values[0]
        print(f"- {disease} (matched symptom: '{symptom_text}')")
    print()
