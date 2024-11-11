import pytesseract
from pdf2image import convert_from_path
import openai
import streamlit as st
from transformers import pipeline
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the NER model
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="bert-large-cased")

def summarize_text_openai(text, max_tokens=100):
    """Summarize text using OpenAI with a token limit."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[ 
            {"role": "system", "content": "You are a summarization assistant for veterinary records."},
            {"role": "user", "content": f"Summarize the following veterinary medical record text in under 100 tokens, preserving critical information: {text}"}
        ],
        temperature=0.5,
        max_tokens=max_tokens
    )
    summary = response['choices'][0]['message']['content']
    return summary

def summarize_large_text(text, max_tokens=100):
    """Summarize large text in chunks while keeping the output under the token limit."""
    chunks = []
    while len(text) > max_tokens:
        split_index = text.rfind("\n", 0, max_tokens)
        if split_index == -1:
            split_index = max_tokens
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    
    summaries = []
    for chunk in chunks:
        summaries.append(summarize_text_openai(chunk, max_tokens=max_tokens))
    return " ".join(summaries)

def generate_recommendations(summary_text, max_tokens=100):
    """Generate recommendations with a token limit of 100."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a veterinary health assistant generating recommendations based on medical records."},
            {"role": "user", "content": f"Provide relevant recommendations and precautions based on the following summary. Limit to 100 tokens: {summary_text}"}
        ],
        temperature=0.5,
        max_tokens=max_tokens
    )
    recommendations = response['choices'][0]['message']['content']
    return recommendations

def extract_entities(text):
    """Extract key entities with Named Entity Recognition (NER)."""
    entities = {
        "Client Info": {"Name": "", "Phone": "", "Address": "", "Email": ""},
        "Patient Info": {"Name": "", "Breed": "", "DOB/Age": "", "Gender": "", "Microchip Number": ""},
        "Vet Clinic Info": {"Visit Date": "", "Clinic Name": "", "Phone": ""},
    }

    ner_results = ner_model(text)
    for result in ner_results:
        label = result['entity']
        entity_text = result['word']

        # Map entities to structured fields
        if label == "B-PER":
            entities["Client Info"]["Name"] = entity_text
        elif label == "B-ORG":
            entities["Vet Clinic Info"]["Clinic Name"] = entity_text
        elif label == "B-LOC":
            entities["Client Info"]["Address"] = entity_text
        # Additional mappings could be added here for other recognized entities.
        
    return entities

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF using OCR."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    # Convert each page to an image
    pages = convert_from_path(temp_pdf_path, 300)

    # Extract text from each page image
    text_data = ''
    for page in pages:
        text = pytesseract.image_to_string(page)
        text_data += text + '\n'

    return text_data

# Streamlit Interface
st.title("Veterinary Health Summary and Recommendation Generator")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Step 1: Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

     # Step 2: Summarize large text in chunks with 100 token output
    summary_text = summarize_large_text(pdf_text, max_tokens=100)
    st.write("### Summarized Text")
    st.write(summary_text)

    # Step 3: Generate Recommendations and Precautions with 100 token output
    recommendations = generate_recommendations(summary_text, max_tokens=100)
    st.write("### Recommendations and Precautions")
    st.write(recommendations)

    # Step 4: Extract Structured Data using NER
    entities = extract_entities(summary_text)
    st.write("### Structured Health Summary")

    st.write("#### Client Information")
    st.write(f"**Name**: {entities['Client Info']['Name']}")
    st.write(f"**Phone**: {entities['Client Info']['Phone']}")
    st.write(f"**Address**: {entities['Client Info']['Address']}")
    st.write(f"**Email**: {entities['Client Info']['Email']}")

    st.write("#### Patient Information")
    st.write(f"**Name**: {entities['Patient Info']['Name']}")
    st.write(f"**Breed**: {entities['Patient Info']['Breed']}")
    st.write(f"**DOB or Age**: {entities['Patient Info']['DOB/Age']}")
    st.write(f"**Gender**: {entities['Patient Info']['Gender']}")
    st.write(f"**Microchip Number**: {entities['Patient Info']['Microchip Number']}")

    st.write("#### Veterinary Clinic Information")
    st.write(f"**Visit Date**: {entities['Vet Clinic Info']['Visit Date']}")
    st.write(f"**Clinic Name**: {entities['Vet Clinic Info']['Clinic Name']}")
    st.write(f"**Phone Number**: {entities['Vet Clinic Info']['Phone']}")

else:
    st.info("Please upload a PDF file to proceed.")
