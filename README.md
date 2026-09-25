# 📝 AI Text Summarizer

An AI-powered **Text Summarization application** that uses **Natural Language Processing (NLP)** and a **T5 Transformer model** to automatically convert long dialogues or text into concise summaries.

The trained T5 model is integrated with a **FastAPI backend** and a simple web interface, allowing users to enter text and generate an AI-based summary.

---

## 🚀 Project Overview

The main goal of this project is to reduce the time required to read long conversations or text by automatically generating a shorter and meaningful summary.

### Workflow

```text
User Input
    ↓
FastAPI Backend
    ↓
Text Cleaning
    ↓
T5 Tokenizer
    ↓
T5 Transformer Model
    ↓
Summary Generation
    ↓
Generated Summary
    ↓
Web Interface

## ✨ Features

- 🤖 AI-based automatic text summarization
- 🧠 T5 Transformer-based summarization
- 🔤 Natural Language Processing (NLP)
- 🧹 Automatic text cleaning and preprocessing
- 🔢 T5 tokenization
- ✂️ Input truncation and padding
- 🎯 Beam search for summary generation
- ⚡ FastAPI REST API
- 🌐 Simple web-based interface
- ✅ Pydantic input validation
- 💻 CPU and GPU support
- 📦 Local trained model integration
- 📚 Interactive FastAPI Swagger documentation
- 🔄 Real-time summary generation

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **T5 Transformer** | AI model used for text summarization |
| **Hugging Face Transformers** | Loading T5 model and tokenizer |
| **PyTorch** | Deep learning framework for model execution |
| **FastAPI** | Backend REST API |
| **Pydantic** | Request data validation |
| **Jinja2** | HTML template rendering |
| **HTML/CSS** | Frontend interface |
| **Regex (`re`)** | Text cleaning and preprocessing |
| **Uvicorn** | ASGI server for running FastAPI |

---
