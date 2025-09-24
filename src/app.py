import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
import folium
from streamlit_folium import st_folium

# --- Page config ---
st.set_page_config(page_title="Healthcare Chatbot", layout="wide")

# --- Load model and dataset ---
model = SentenceTransformer('models/all-MiniLM-L6-v2')
df = pd.read_csv("../data/symptoms_diseases.csv")
df['symptoms_text'] = df['symptoms'].astype(str)
symptom_texts = df['symptoms_text'].tolist()

# --- Doctors and Hospitals ---
doctor_dict = {
    'cold': ['General Physician', 'ENT Specialist'],
    'flu': ['General Physician', 'Infectious Disease Specialist'],
    'migraine': ['Neurologist']
}
hospitals = {
    'General Hospital': (28.6139, 77.2090),
    'City Clinic': (28.6200, 77.2100),
    'Sunrise Hospital': (28.6150, 77.2150)
}
health_tips = {
    "flu": ["Drink plenty of fluids", "Rest well", "Take paracetamol if fever persists"],
    "cold": ["Warm water gargle", "Stay hydrated", "Avoid cold drinks"],
    "migraine": ["Reduce screen time", "Avoid triggers like strong light or smell", "Rest in a dark room"]
}

# --- Session state for chat ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'predicted_disease' not in st.session_state:
    st.session_state['predicted_disease'] = None

# --- App Header ---
st.markdown("""
<div style='background-color:#4CAF50;padding:10px;border-radius:10px;text-align:center'>
<h1 style='color:white;'>🩺 Healthcare Chatbot</h1>
<p style='color:white;'>Your virtual assistant for health & wellness</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- User Input ---
user_input = st.text_input("Describe your symptoms:")

if st.button("Submit") and user_input:
    match, score, idx = process.extractOne(user_input, symptom_texts, scorer=fuzz.WRatio, score_cutoff=60)
    if match:
        predicted_disease = df.iloc[idx]['disease']
        st.session_state['predicted_disease'] = predicted_disease
        st.session_state['chat_history'].append(("You", user_input))
        st.session_state['chat_history'].append(("Bot", f"Predicted Disease: **{predicted_disease}** ✅"))
    else:
        st.session_state['chat_history'].append(("You", user_input))
        st.session_state['chat_history'].append(("Bot", "No matching disease found. Please describe your symptoms more clearly."))

# --- Display Chat History ---
for speaker, message in st.session_state['chat_history']:
    if speaker == "You":
        st.markdown(f"<div style='text-align:right;background-color:#D3F8E2;padding:8px;border-radius:10px;margin:5px'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;background-color:#F1F0F0;padding:8px;border-radius:10px;margin:5px'>{message}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["💊 Predicted Disease", "👨‍⚕️ Suggested Doctors", "🏥 Nearby Hospitals", "💡 Health Tips & FAQs"])

with tab1:
    st.subheader("Disease Prediction")
    disease = st.session_state['predicted_disease']
    if disease:
        st.markdown(f"""
        <div style="background-color:#FFE6E6;padding:10px;border-radius:10px;">
        <h3>🦠 {disease}</h3>
        <p>Summary about {disease}. Severity, symptoms, and basic info can go here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Enter symptoms to get a disease prediction.")

with tab2:
    st.subheader("Doctors You Can Consult")
    if disease:
        docs = doctor_dict.get(disease.lower(), ['General Physician'])
        for doc in docs:
            st.markdown(f"""
            <div style="border:1px solid #FF5733; padding:10px; border-radius:10px; margin-bottom:5px; background-color:#FFF0F5">
            <h4>🩺 {doc}</h4>
            <p>Specialist for {disease}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Predict a disease first to see doctor suggestions.")

with tab3:
    st.subheader("Nearby Hospitals")
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=13)
    for name, coords in hospitals.items():
        folium.Marker(location=coords, popup=name, icon=folium.Icon(color='red')).add_to(m)
    st_folium(m, width=700, height=400)

with tab4:
    st.subheader("Health Tips & FAQs")
    if disease:
        tips = health_tips.get(disease.lower(), [])
        for tip in tips:
            st.markdown(f"<div style='background-color:#E6F7FF;padding:8px;border-radius:10px;margin-bottom:5px'>💡 {tip}</div>", unsafe_allow_html=True)
    st.markdown("""
    <details>
    <summary>Frequently Asked Questions</summary>
    <ul>
        <li>Q: How to prevent flu? <br> A: Get vaccinated, wash hands frequently, stay hydrated.</li>
        <li>Q: When to see a doctor? <br> A: If symptoms persist for more than 3 days or worsen.</li>
        <li>Q: Can cold turn into flu? <br> A: Usually no, but complications may occur in weak immunity.</li>
    </ul>
    </details>
    """, unsafe_allow_html=True)
