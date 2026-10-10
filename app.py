
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).parent / "sample_documents"
FALLBACK = "Information not found in provided document"


def load_and_chunk_documents(folder, max_words=100):
    chunks = []

    for file_path in sorted(Path(folder).glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]

        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                proposed = " ".join(current + [sentence])

                if current and len(proposed.split()) > max_words:
                    chunks.append({
                        "source": file_path.name,
                        "text": " ".join(current)
                    })
                    current = [sentence]
                else:
                    current.append(sentence)

            if current:
                chunks.append({
                    "source": file_path.name,
                    "text": " ".join(current)
                })

    return chunks


class LocalQAAssistant:
    def __init__(self, chunks, threshold=0.15, top_k=3):
        if not chunks:
            raise ValueError("No document chunks found.")

        self.chunks = chunks
        self.threshold = threshold
        self.top_k = top_k

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.document_vectors = self.vectorizer.fit_transform(
            [item["text"] for item in chunks]
        )

    def answer_query(self, query):
        if not query.strip():
            return {
                "answer": "Please enter a question.",
                "source": None,
                "passage": None,
                "score": 0.0,
                "retrieved": []
            }

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector, self.document_vectors
        ).flatten()

        indices = np.argsort(scores)[::-1][:self.top_k]

        retrieved = [
            {
                "source": self.chunks[i]["source"],
                "passage": self.chunks[i]["text"],
                "score": float(scores[i])
            }
            for i in indices
        ]

        if not retrieved or retrieved[0]["score"] < self.threshold:
            return {
                "answer": FALLBACK,
                "source": None,
                "passage": None,
                "score": retrieved[0]["score"] if retrieved else 0.0,
                "retrieved": retrieved
            }

        best = retrieved[0]

        return {
            "answer": best["passage"],
            "source": best["source"],
            "passage": best["passage"],
            "score": best["score"],
            "retrieved": retrieved
        }


def main():
    chunks = load_and_chunk_documents(DATA_DIR)
    assistant = LocalQAAssistant(chunks)

    print("Local QA Assistant")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        result = assistant.answer_query(question)

        print("\nAnswer:", result["answer"])

        if result["passage"]:
            print("Source document:", result["source"])
            print("Supporting passage:", result["passage"])

        print(f"Retrieval score: {result['score']:.4f}")

        if result["retrieved"]:
            print("\nTop retrieved passages:")
            for item in result["retrieved"]:
                print(
                    f"- {item['source']} "
                    f"(score: {item['score']:.4f})"
                )
                print(" ", item["passage"])


if __name__ == "__main__":
    main()
