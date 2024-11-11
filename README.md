# Veterinary Health Summary and Recommendation Generator

This project is a veterinary health assistant built using Streamlit, OpenAI's GPT-4 API, and Named Entity Recognition (NER) to extract, summarize, and generate recommendations from veterinary medical records in PDF format. The system processes veterinary documents, extracts key information such as client details, patient information, and clinic info, and generates summaries and recommendations.

## Features

- **PDF Upload & Text Extraction**: Extracts text from uploaded veterinary PDFs using OCR (Optical Character Recognition) through Tesseract.
- **Text Summarization**: Summarizes long veterinary medical records using OpenAI's GPT-4 model.
- **Named Entity Recognition (NER)**: Extracts important entities such as client details, patient info, and clinic info from the veterinary medical record.
- **Recommendation Generation**: Generates relevant health recommendations and precautions based on the medical record.

## Installation

To get started with the project, follow the steps below to set it up locally.

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- `pip` (Python's package installer)

### Steps to Install

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/veterinary-health-summary.git
   cd veterinary-health-summary

Here is the full README.md for your project, compiled all together:

markdown
Copy code
# Veterinary Health Summary and Recommendation Generator

This project is a veterinary health assistant built using Streamlit, OpenAI's GPT-4 API, and Named Entity Recognition (NER) to extract, summarize, and generate recommendations from veterinary medical records in PDF format. The system processes veterinary documents, extracts key information such as client details, patient information, and clinic info, and generates summaries and recommendations.

## Features

- **PDF Upload & Text Extraction**: Extracts text from uploaded veterinary PDFs using OCR (Optical Character Recognition) through Tesseract.
- **Text Summarization**: Summarizes long veterinary medical records using OpenAI's GPT-4 model.
- **Named Entity Recognition (NER)**: Extracts important entities such as client details, patient info, and clinic info from the veterinary medical record.
- **Recommendation Generation**: Generates relevant health recommendations and precautions based on the medical record.

## Installation

To get started with the project, follow the steps below to set it up locally.

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- `pip` (Python's package installer)

### Steps to Install

**Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/veterinary-health-summary.git
   cd veterinary-health-summary
Create and Activate a Virtual Environment (optional but recommended):


## Install Dependencies
bash
Copy code
pip install -r requirements.txt

## Set Up API Keys:
Create a .env file in the project root and add OpenAI API key:

OPENAI_API_KEY=your-openai-api-key-here

## Dependencies
This project relies on the following libraries:

streamlit: For building the interactive web application.
openai: For interacting with OpenAI's GPT-4 model to generate summaries and recommendations.
transformers: For using pre-trained models like BERT for Named Entity Recognition (NER).
pdf2image: To convert PDF pages into images for OCR processing.
pytesseract: To perform OCR and extract text from images.
python-dotenv: To load environment variables from a .env file.

Install all dependencies with:
pip install -r requirements.txt

Run the Streamlit Application: After setting up the project, you can run the Streamlit app with the following command:
streamlit run app.py
