import argparse
import os
import pickle
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

parser = argparse.ArgumentParser(description='Strikingly Support Q&A')
parser.add_argument('question', type=str, help='Your question for Strikingly Support')
args = parser.parse_args()

try:
    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
except FileNotFoundError:
    print("Error: faiss_store.pkl not found. Please ensure the vectorstore file exists.")
    exit(1)

def get_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

def retrieve_documents(query, k=3):
    docs = store.similarity_search(query, k=k)
    return docs

def build_prompt(query, documents):
    """Build Q&A prompts"""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = f"""
    You are a helpful support assistant. Answer the user's question based on the following context:

    {context}

    Question: {query}
    Answer:
    """
    return prompt

def main():
    documents = retrieve_documents(args.question)

    if not documents:
        print("Sorry, I couldn't find any relevant information. Please try another question.")
        return

    prompt = build_prompt(args.question, documents)

    answer = get_chat_response(prompt)

    print(f" Answer:\n{answer}\n\n Sources:\n{', '.join([doc.metadata['source'] for doc in documents])}")


if __name__ == "__main__":
    main()