import streamlit as st
import fitz  # PyMuPDF
import nltk
import os
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 GenAI Assistant for Research Summarization (FREE)")

uploaded_file = st.file_uploader("📤 Upload PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    def extract_text(file):
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        else:
            text = file.read().decode("utf-8")
        return text

    def summarize(text):
        chunks = [text[i:i+1000] for i in range(0, min(len(text), 3000), 1000)]
        summaries = summarizer(chunks, max_length=100, min_length=30, do_sample=False)
        return " ".join([s['summary_text'] for s in summaries])[:150] + "..."

    def get_most_relevant_paragraph(question, paragraphs):
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        para_embeddings = embedder.encode(paragraphs, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, para_embeddings)[0]
        best_idx = scores.argmax().item()
        return paragraphs[best_idx], best_idx

    def generate_questions(text, num=3):
        sentences = sent_tokenize(text)
        questions = []
        for _ in range(num):
            sent = random.choice(sentences)
            words = sent.split()
            if len(words) > 6:
                answer = random.choice(words)
                question = sent.replace(answer, "_____")
                questions.append((question, answer, sent))
        return questions

    full_text = extract_text(uploaded_file)
    paragraphs = [p for p in full_text.split("\n") if len(p.strip()) > 30]

    st.subheader("📌 Auto Summary")
    st.info(summarize(full_text))

    mode = st.radio("Choose interaction mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        user_question = st.text_input("❓ Ask a question based on the document:")
        if user_question:
            paragraph, para_idx = get_most_relevant_paragraph(user_question, paragraphs)
            result = qa_pipeline(question=user_question, context=paragraph)
            st.write(f"**Answer:** {result['answer']}")
            st.caption(f"📖 Justified from paragraph #{para_idx + 1}: “{paragraph[:150]}...”")

    elif mode == "Challenge Me":
        st.markdown("🤖 Generating 3 questions from the document...")
        questions = generate_questions(full_text)

        user_answers = []
        for i, (q, ans, ref) in enumerate(questions):
            user_input = st.text_input(f"Q{i+1}: {q}")
            user_answers.append((user_input, ans, ref))

        if all(u[0] for u in user_answers):
            st.subheader("🧾 Evaluation & Feedback")
            for i, (user_ans, correct_ans, ref_text) in enumerate(user_answers):
                st.markdown(f"**Q{i+1}:** Your Answer: `{user_ans}`")
                if user_ans.lower().strip() == correct_ans.lower().strip():
                    st.success(f"✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Correct Answer: `{correct_ans}`")
                st.caption(f"📖 Based on: “{ref_text[:150]}...”")

