# CREA: Clinical Record Entity Analyzer
**A State-of-the-Art Deep Learning Model for Healthcare Information Extraction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![F1-Score](https://img.shields.io/badge/F1--Score-87.23%25-brightgreen.svg)](https://en.wikipedia.org/wiki/F-score)

---

<!--## 🚀 Project Demo

<p align="center">
  <img src="demo.gif" alt="CREA Project Demo GIF" width="80%">
</p>

*(This project also includes a `Gradio` script to deploy this model as an interactive web app.)*

---- !-->

## 1. 🎯 The Problem
In the healthcare and life sciences industry, over 80% of valuable data is **"unstructured"**—locked away in doctor's notes, clinical trial reports, and research papers. This data is critical, but it's also messy and impossible to query at scale.

**The Business Case:**
Imagine a pharmaceutical company needing to analyze 10 million patient records. They want to answer a simple question:

> "How many patients with **acute renal failure** were also prescribed **paracetamol**?"

Answering this requires manually reading millions of documents, a process that is slow, expensive, and prone to human error.

---

## 2. 💡 The Solution: CREA
**CREA (Clinical Record Entity Analyzer)** is a fine-tuned BioBERT model that reads unstructured clinical text and automatically extracts and labels key pieces of information. It acts as a high-speed "information extractor," turning messy sentences into structured, queryable data.

When evaluated on the unseen test set of the **BC5CDR (BioCreative V Chemical Disease Relation) benchmark**, the model achieved:

| Metric | Score |
| :--- | :--- |
| **Macro F1-Score** | **87.23%** |
| **Test Loss** | **0.0990** |

An F1-score of this level is considered **state-of-the-art** for this task, proving the model's high accuracy and reliability in a complex domain.

---

## 3. 🧠 Technical Deep-Dive

This project uses a **Transformer** model, the same architecture behind modern LLMs. Here’s a breakdown of the key technical decisions.

### 3.1. The Data: BC5CDR Corpus
The model was trained on the **BC5CDR dataset**, a collection of 1500 PubMed abstracts. The text is annotated in the **IOB (Inside, Outside, Beginning)** format, which labels every single word:

| Word | Tag | Explanation |
| :--- | :--- | :--- |
| "patient" | `O` | (Outside an entity) |
| "has" | `O` | (Outside an entity) |
| "type" | `B-Disease` | (Beginning of a Disease) |
| "2" | `I-Disease` | (Inside of a Disease) |
| "diabetes"| `I-Disease` | (Inside of a Disease) |

### 3.2. The Model: Why BioBERT?
A standard `BERT` model is trained on Wikipedia and books. It has no knowledge of medical jargon. Using it would be like asking a literature major to read a medical chart.

Instead, this project uses **`dmis-lab/biobert-base-cased-v1.1`**. This is a BERT model that was pre-trained from scratch on **millions of PubMed abstracts**. It's a *domain expert* that already understands the language of medicine, which is why it can achieve such high accuracy. We then **fine-tuned** this expert model on our specific task.

### 3.3. The "Gotcha": Aligning Sub-word Tokens
The hardest part of NER with transformers is the tokenizer. A tokenizer splits words into "sub-words."

* **Problem:** The word `hypotension` becomes two tokens: `['hypo', '##tension']`. But we only have *one* label (`B-Disease`) for the original word.
* **Solution:** We must align the labels. We assign the true label `B-Disease` to the *first* sub-token (`'hypo'`) and a special "ignore" index (`-100`) to all other sub-tokens (`'##tension'`). This teaches the model to only make predictions at the start of each new word, drastically improving stability.

---

