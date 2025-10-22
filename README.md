

-----

```markdown
# CREA: Clinical Record Entity Analyzer
**A State-of-the-Art Deep Learning Model for Healthcare Information Extraction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![F1-Score](https://img.shields.io/badge/F1--Score-87.23%25-brightgreen.svg)](https://en.wikipedia.org/wiki/F-score)

---

## 1. Project Overview & Business Problem

In the healthcare and life sciences industry, over 80% of valuable data is "unstructured" — locked away in doctor's notes, clinical trial reports, patient histories, and research papers. This data is critical, but it's also messy and impossible to query at scale.

**The Business Problem:**
Imagine a pharmaceutical company (like a ZS client) needing to analyze 10 million patient records. They want to answer a simple question:

> "How many patients with **acute renal failure** were also prescribed **paracetamol**?"

Answering this requires manually reading millions of documents, a process that is slow, expensive, and prone to human error.

**The Solution: CREA (Clinical Record Entity Analyzer)**
This project is a deep learning model that reads unstructured clinical text and automatically extracts and labels key pieces of information. It acts as an "information extractor," turning messy sentences into structured, queryable data.

This project was built to solve this exact challenge, directly aligning with the advanced analytics and healthcare focus of ZS Associates.

---

## 2. Performance & Results

This model isn't just a prototype; it's a high-performance, research-grade tool.

When evaluated on the unseen test set of the **BC5CDR (BioCreative V Chemical Disease Relation) benchmark**, the model achieved:

| Metric | Score |
| :--- | :--- |
| **Macro F1-Score** | **87.23%** |
| **Test Loss** | **0.0990** |

An F1-score of this level is considered **state-of-the-art** for this task, proving the model's high accuracy and reliability.

### 🚀 Live Demo

Here is a real example of the model's output:

**Input Text:**
> "A case of metabolic acidosis and acute renal failure following paracetamol ingestion is presented."

**Extracted Entities (Output):**
```

  - Entity: Disease
    Word: metabolic acidosis
    Score: 0.9960

  - Entity: Disease
    Word: acute renal failure
    Score: 0.9977

  - Entity: Chemical
    Word: paracetamol
    Score: 0.9994

<!-- end list -->

````

---

## 3. Technical Deep-Dive

This project uses a **Transformer** model, the same architecture behind modern marvels like ChatGPT. Here’s a breakdown of the key technical decisions.

### 3.1. The Data: BC5CDR Corpus

The model was trained on the **BC5CDR dataset**, a collection of 1500 PubMed abstracts. The text is annotated in the **IOB (Inside, Outside, Beginning)** format, which labels every single word:

| Word | Tag | Explanation |
| :--- | :--- | :--- |
| "patient" | `O` | (Outside an entity) |
| "has" | `O` | (Outside an entity) |
| "type" | `B-Disease` | (Beginning of a Disease) |
| "2" | `I-Disease` | (Inside of a Disease) |
| "diabetes" | `I-Disease` | (Inside of a Disease) |

### 3.2. The Model: Why BioBERT?

A standard `BERT` model is trained on Wikipedia and books. It has no knowledge of medical jargon. Using it would be like asking a literature major to read a medical chart.

Instead, this project uses **`dmis-lab/biobert-base-cased-v1.1`**. This is a BERT model that was pre-trained from scratch on **millions of PubMed abstracts**. It's a *domain expert* that already understands the language of medicine, which is why it can achieve such high accuracy. We then **fine-tuned** this expert model on our specific task.

### 3.3. The "Gotcha": Aligning Sub-word Tokens

The hardest part of NER with transformers is the tokenizer. A tokenizer splits words into "sub-words."

* **Problem:** The word `hypotension` becomes two tokens: `['hypo', '##tension']`. But we only have *one* label (`B-Disease`) for the original word.
* **Solution:** We must align the labels. We assign the true label `B-Disease` to the *first* sub-token (`'hypo'`) and a special "ignore" index (`-100`) to all other sub-tokens (`'##tension'`). This teaches the model to only make predictions at the start of each new word, drastically improving stability and accuracy.

---

## 4. How to Use This Model

The fine-tuned model is saved in this repository. You can load and use it for inference in just a few lines.

### 4.1. Prerequisites

First, install the necessary libraries:
```bash
pip install transformers torch
````

### 4.2. Run Inference in Python

This script will load the saved model from this repository and run a prediction.

```python
from transformers import pipeline
import os

# --- 1. Helper Function to Clean Output ---
# This function groups sub-words (e.g., "hypo", "##tension" -> "hypotension")
def post_process_results(results):
    entities = []
    current_entity = None
    for res in results:
        entity_tag = res['entity'].split('-')[-1]
        if res['entity'].startswith('B-'):
            if current_entity: entities.append(current_entity)
            current_entity = {"entity": entity_tag, "word": res['word'], "score": res['score']}
        elif res['entity'].startswith('I-') and current_entity:
            if res['word'].startswith('##'):
                current_entity['word'] += res['word'].replace('##', '')
            else:
                current_entity['word'] += ' ' + res['word']
        else:
            if current_entity: entities.append(current_entity)
            current_entity = None
    if current_entity: entities.append(current_entity)
    return entities

# --- 2. Load and Use the Model ---

# Define the path to your saved model (the final checkpoint)
MODEL_PATH = "./biobert-ner-bc5cdr/checkpoint-1710"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please make sure you have downloaded the saved model to this directory.")
else:
    print("Loading model...")
    # Load the fine-tuned model into a pipeline
    # Use device=0 for GPU or device=-1 for CPU
    ner_pipeline = pipeline("ner", model=MODEL_PATH, tokenizer=MODEL_PATH, device=0)
    print("🚀 Model loaded successfully!")

    # --- 3. Run Inference ---
    text = "The patient was prescribed Selegiline for Parkinson's disease, but developed severe hypotension."

    raw_results = ner_pipeline(text)
    clean_results = post_process_results(raw_results)

    print(f"\nInput: {text}")
    print("Extracted Entities:")
    for entity in clean_results:
        print(f"  - {entity['word']} ({entity['entity']}) | Confidence: {entity['score']:.4f}")

```

-----

## 5\. Repository Structure

```
.
├── biobert-ner-bc5cdr/
│   ├── checkpoint-1710/      <-- THIS IS THE FINAL TRAINED MODEL
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   ├── vocab.txt
│   │   └── ...
│   └── ...
├── data/
│   ├── train.tsv
│   ├── devel.tsv
│   ├── test.tsv
├── CREA_Training_Notebook.ipynb  <-- The full Google Colab training script
└── README.md                     <-- You are here
```


```
