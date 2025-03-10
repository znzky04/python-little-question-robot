import argparse
import pickle
import requests
import xmltodict
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())


def get_articles_from_zendesk(zendesk_url):
    response = requests.get(zendesk_url)
    articles = response.json().get('articles', [])
    return [{"text": clean_html(article['body']), "source": article['html_url']} for article in articles]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding ？？？ support')
    # parser.add_argument('-m', '--mode', type=str, choices=['sitemap', 'zendesk'], default='sitemap',
    #                     help='Mode for data extraction: sitemap or zendesk')
    # parser.add_argument('-s', '--sitemap', type=str, required=False,
    #                     help='URL to your sitemap.xml', default='https://www.paepper.com/sitemap.xml')
    # parser.add_argument('-f', '--filter', type=str, required=False,
    #                     help='Text which needs to be included in all URLs which should be considered',
    #                     default='https://www.paepper.com/blog/posts')
    parser.add_argument('-z', '--zendesk', type=str, required=True,
                        help='https://？？？/api-reference/help_center/help-center-api/articles/#list-articles')
    args = parser.parse_args()

    articles = get_articles_from_zendesk(args.zendesk)

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for article in articles:
        splits = text_splitter.split_text(article['text'])
        docs.extend(splits)
        metadatas.extend([{"source": article['source']}] * len(splits))
        print(f"Split {article['source']} into {len(splits)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key="自己填api")
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
