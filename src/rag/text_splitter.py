from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    loader = TextLoader("./src/data/notes.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Number of chunks: {len(chunks)}")
    print(chunks)

if __name__ == "__main__":
    main()