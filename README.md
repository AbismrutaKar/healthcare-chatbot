# Healthcare Chatbot

A smart healthcare chatbot that predicts possible diseases based on user symptoms, suggests doctors and nearby hospitals, and provides medical information. Built using machine learning and natural language processing.

---

## Features

- Predict diseases from user-input symptoms
- Suggest relevant doctors and nearby hospitals
- Provide FAQ responses related to common health queries
- Interactive web app using Streamlit

---

## Technologies Used

- Python
- Pandas, NumPy
- TensorFlow / Keras
- Sentence Transformers (for symptom embeddings)
- Streamlit (for web interface)
- Folium (for interactive maps)
- RapidFuzz (for fuzzy matching)
- Git & GitHub (for version control)

---

## Dataset

- `symptoms_diseases.csv`: Contains symptom-disease pairs and optional FAQs
- Expanded to include 50–100+ rows for better predictions

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/healthcare-chatbot.git
cd healthcare-chatbot/src
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Make sure the models folder contains the trained models:

```
../models/
```

---

## Usage

1. Run the chatbot in the terminal:

```bash
python predict.py
```

2. Run the Streamlit web app:

```bash
streamlit run app.py
```

3. Interact with the chatbot:

- Enter symptoms to get predicted diseases
- View suggested doctors and hospitals
- Access FAQ answers

---

## Project Structure

```
healthcare-chatbot/
│
├─ src/
│   ├─ app.py
│   ├─ train.py
│   ├─ predict.py
│   └─ requirements.txt
│
├─ data/
│   └─ symptoms_diseases.csv
│
├─ models/
│   └─ (trained ML models)
│
└─ README.md
```

---

## Future Enhancements

- Add more disease prediction accuracy by expanding the dataset
- Integrate real-time hospital API for doctor/hospital suggestions
- Add voice input for symptoms
- Add symptom severity scoring for more precise predictions

---

## License

This project is open source and available under the MIT License.

---

## Contact

- GitHub: [https://github.com/<your-username>](https://github.com/<your-username>)
- Email: <your-email@example.com>