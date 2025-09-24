# preprocess.py

import re

def clean_text(text):
    # lowercase and remove special characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def symptoms_text_from_row(row):
    # row is now a string from the 'symptoms' column
    return clean_text(row)
