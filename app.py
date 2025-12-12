import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from ingest import run_ingestion
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec



load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=PINECONE_API_KEY)


@app.route("/")
def home():
    return {"status": "DocuMind API running"}


@app.route("/ingest", methods=["POST"])
def ingest():
    result = run_ingestion()
    return jsonify({"message": result})


@app.route("/query", methods=["GET"])
def query_data():
    question = request.args.get("q")
    if not question:
        return jsonify({"error": "Missing ?q=question"}), 400

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vec = embeddings.embed_query(question)

    index = pc.Index(PINECONE_INDEX)

    result = index.query(
        vector=vec,
        top_k=3,
        include_metadata=True
    )

    return jsonify(result.to_dict())


if __name__ == "__main__":
    app.run(port=5000, debug=True)
