# 🧠 CareerLens AI

**AI-Powered Resume & Job Matching Platform**

CareerLens AI is an intelligent resume analysis application that compares a candidate's resume with a job description using Natural Language Processing (NLP), multilingual semantic similarity, and skill matching.

The application helps candidates understand how well their resume matches a job and identifies skills that may need improvement.

## 🚀 Features

- 📄 PDF resume text extraction
- 🎯 Resume-to-job skill matching
- 🌐 Arabic and English job description support
- 🧠 AI-powered multilingual semantic similarity
- 📊 Overall job match score
- 📈 ATS readiness analysis
- ✅ Matched skills detection
- 🟡 Related skills detection
- ❌ Missing skills detection
- 💡 Smart resume recommendations
- 📋 Detailed score breakdown
- 🌙 Modern Streamlit interface

## 🧠 How It Works

CareerLens AI analyzes a resume and job description through multiple components:

1. Extracts text from the uploaded PDF resume.
2. Detects technical and professional skills.
3. Compares resume skills with job requirements.
4. Uses a multilingual Sentence Transformer model to measure semantic similarity between the resume and job description.
5. Evaluates resume structure for ATS readiness.
6. Generates skill-gap insights and personalized recommendations.

### Overall Match

The overall match score combines:

- **60% Skill Matching**
- **40% Semantic Similarity**

This provides both keyword/skill-based and AI-powered contextual analysis.

## 🛠️ Technologies

- Python
- Streamlit
- Sentence Transformers
- Natural Language Processing (NLP)
- Hugging Face
- PyPDF
- Git
- GitHub

## 🤖 AI Model

CareerLens AI uses:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

The multilingual model enables semantic comparison between resumes and job descriptions written in different languages, including **Arabic and English**.

## 📊 Analysis Results

The application provides:

- Overall Match Score
- Skills Match Score
- Semantic AI Score
- ATS Readiness Score
- Score Breakdown
- Matched Skills
- Related Skills
- Missing Skills
- Resume Structure Analysis
- Smart Recommendations

## 💻 Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## ⚠️ Disclaimer

CareerLens AI provides estimated resume and job-match insights for informational purposes. The ATS readiness score is not an official score from an employer or Applicant Tracking System.

## 👨‍💻 Developer

**Mohmmad Nhari**

Computer Science Graduate  
Interested in Artificial Intelligence, Software Development, and intelligent applications.