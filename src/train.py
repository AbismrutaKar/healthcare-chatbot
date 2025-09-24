import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text, symptoms_text_from_row

# 1. Load data
df = pd.read_csv("../data/symptoms_diseases.csv")
df['symptoms_text'] = df['symptoms'].apply(symptoms_text_from_row)
df['symptoms_text_clean'] = df['symptoms_text'].apply(clean_text)

# 2. Embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast

# 3. Disease model: KNN
X = embedder.encode(df['symptoms_text_clean'].tolist(), convert_to_numpy=True)
y = df['disease'].tolist()
disease_model = KNeighborsClassifier(n_neighbors=1, metric='cosine')
disease_model.fit(X, y)

# Evaluate quickly
preds = disease_model.predict(X)
print("Disease model accuracy:", accuracy_score(y, preds))
print(classification_report(y, preds))

# 4. Intent classifier (symptom vs FAQ)
intent_texts = [
    "I have a headache and nausea", 
    "My nose is runny and I cough",   # symptom
    "How to reduce fever?", 
    "When to see a doctor for headache?"  # faq
]
intent_labels = ["symptom", "symptom", "faq", "faq"]

intent_texts_clean = [clean_text(t) for t in intent_texts]
X_intent = embedder.encode(intent_texts_clean, convert_to_numpy=True)
intent_clf = LogisticRegression(max_iter=1000)
intent_clf.fit(X_intent, intent_labels)

print("Intent predictions:", intent_clf.predict(X_intent))

# 5. Save models
joblib.dump(intent_clf, "../models/intent_model.joblib")
joblib.dump(disease_model, "../models/disease_model.joblib")
print("✅ Models saved in ../models/")
