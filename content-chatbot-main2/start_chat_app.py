import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import pickle

_condense_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

# Define QA_PROMPT
_qa_template = """You are a helpful support assistant. Answer the user's question based on the following context:
{context}

Question: {question}
Answer:"""
QA_PROMPT = PromptTemplate(template=_qa_template, input_variables=["context", "question"])

parser = argparse.ArgumentParser(description="？？？ Support Chat")
parser.add_argument("question", type=str, help="Your question for ？？？ Support")
args = parser.parse_args()

try:
    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)
except FileNotFoundError:
    print("Error: faiss_store.pkl not found. Please ensure the vectorstore file exists.")
    exit(1)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=store.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},  # Use QA_PROMPT
    return_source_documents=True  # Ensure returning source documents
)

def main():
    result = chain({"question": args.question, "chat_history": []})

    sources = list(set([doc.metadata['source'] for doc in result['source_documents']]))

    print(f" Answer:\n{result['answer']}\n\n Sources:\n{', '.join(sources)}")

if __name__ == "__main__":
    main()
