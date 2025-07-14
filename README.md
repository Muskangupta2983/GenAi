# 📄 Auto Summary – GenAI Document Summarizer & Q&A App

A Streamlit-based Generative AI project that summarizes PDF documents and allows users to ask questions based on the content using state-of-the-art NLP models like BART and Sentence Transformers.

---

## 🚀 Features

- 🔍 **Automatic Document Summarization**
  - Uses HuggingFace Transformers (`facebook/bart-large-cnn`) to generate concise summaries of uploaded PDFs.

- 🧠 **Ask Questions Based on Document**
  - Users can query the document using natural language.
  - Uses `sentence-transformers` to find the most relevant answer from the PDF.

- 🎯 **Two Modes of Interaction**
  - **Ask Anything**: Ask any question about the document.
  - **Challenge Me**: (Optional future feature for quiz-style interaction.)

- 💬 **Justified Answers**
  - Provides evidence from the original document with paragraph reference.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **NLP Models:** 
  - Summarization: `facebook/bart-large-cnn`
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- **Libraries:** `transformers`, `nltk`, `PyMuPDF`, `sentence-transformers`

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/auto-summary-app.git
cd auto-summary-app

# Create Environment
conda create -n auto-summary python=3.10
conda activate auto-summary

# Install Dependencies
pip install -r requirements.txt
Or
pip install streamlit PyMuPDF nltk transformers sentence-transformers


###How to run
streamlit run app.py
