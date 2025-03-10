import argparse
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

parser = argparse.ArgumentParser(description='？？？ Support Q&A')
parser.add_argument('question', type=str, help='Your question for ？？？ Support')
args = parser.parse_args()

try:
    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
except FileNotFoundError:
    print("Error: faiss_store.pkl not found. Please ensure the vectorstore file exists.")
    exit(1)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=store.as_retriever(),
    return_source_documents=True
)

def main():
    try:
        result = qa_chain({"question": args.question, "chat_history": []})

        print(f"Answer:\n{result['answer']}\n")

        sources = set([doc.metadata['source'] for doc in result['source_documents']])
        if sources:
            print("Sources:")
            for source in sources:
                print(f"- [{source}]({source})")
        else:
            print("No relevant sources found.")

    except Exception as e:
        print(f"An error occurred while processing your question: {e}")


if __name__ == "__main__":
    main()

