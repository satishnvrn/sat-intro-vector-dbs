import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingestion Started...")
    loader = UnstructuredHTMLLoader("./data/blender_manual_v420_en.html/index.html")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print("splitting done...")
    print(f"Number of texts: {len(texts)}")

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
