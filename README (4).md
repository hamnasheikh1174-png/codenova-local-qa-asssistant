# CodeNova Task 4: Local Question-Answering Assistant

**Author:** Hamna Zahoor

## Project Overview

A local Python Question-Answering Assistant that retrieves relevant information from text documents using TF-IDF and cosine similarity. It displays source passages and handles unsupported questions without paid APIs.

## Technologies

- Python
- Scikit-learn
- TF-IDF and Cosine Similarity

## Supported Test Questions

1. What does CodeNova provide?
2. What are CodeNova's internship tasks?
3. What skills can students develop during an internship at CodeNova?
4. What is Machine Learning?
5. What is supervised learning?
6. What is TF-IDF?
7. What is RAG?
8. What are common regression metrics?

## Unsupported Test Questions

1. What is the capital of France?
2. Who won the FIFA World Cup in 1998?

**Expected fallback:**
`Information not found in provided document`

## Features

- Reads local text documents.
- Retrieves relevant passages.
- Displays source filenames and retrieval scores.
- Returns a fallback for unsupported questions.

## How to Run

1. Install dependencies using `pip install -r requirements.txt`.
2. Run `python app.py`.
3. Enter questions and review the answers and source passages.
4. Type `exit` to close the assistant.

## Conclusion

This project demonstrates local document retrieval, source-based answers, and unsupported-question handling without paid APIs.
