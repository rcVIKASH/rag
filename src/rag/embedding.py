from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def main():
    documents = [
        Document(
            page_content=(
                "Rajasthan is the largest state in India by area. Jaipur, "
                "known as the Pink City, is its capital. The state is famous "
                "for its deserts, forts, and rich cultural heritage."
            ),
            metadata={"region": "West", "state": "Rajasthan"},
        ),
        Document(
            page_content=(
                "Gujarat is known for the Statue of Unity, Gir National Park, "
                "and a strong industrial economy. Gandhinagar is its capital."
            ),
            metadata={"region": "West", "state": "Gujarat"},
        ),
        Document(
            page_content=(
                "Punjab is called the Land of Five Rivers. Chandigarh serves "
                "as its capital, and the state is famous for agriculture and "
                "Sikh heritage."
            ),
            metadata={"region": "North", "state": "Punjab"},
        ),
        Document(
            page_content=(
                "Himachal Pradesh is known for the Himalayas, hill stations "
                "like Shimla and Manali, and apple orchards. Shimla is the capital."
            ),
            metadata={"region": "North", "state": "Himachal Pradesh"},
        ),
        Document(
            page_content=(
                "Tamil Nadu is famous for its ancient temples, Bharatanatyam "
                "dance, and the city of Chennai, which is the state capital."
            ),
            metadata={"region": "South", "state": "Tamil Nadu"},
        ),
        Document(
            page_content=(
                "Karnataka is home to Bengaluru, India's technology hub. "
                "The state is also known for Mysore Palace and Hampi."
            ),
            metadata={"region": "South", "state": "Karnataka"},
        ),
        Document(
            page_content=(
                "West Bengal has Kolkata as its capital and is famous for "
                "Durga Puja, the Sundarbans, and its literary and cultural traditions."
            ),
            metadata={"region": "East", "state": "West Bengal"},
        ),
        Document(
            page_content=(
                "Odisha is known for the Jagannath Temple in Puri, Konark Sun "
                "Temple, and beautiful beaches. Bhubaneswar is the capital."
            ),
            metadata={"region": "East", "state": "Odisha"},
        ),
        Document(
            page_content=(
                "Assam is famous for its tea gardens, Kaziranga National Park, "
                "and the mighty Brahmaputra River. Dispur is the capital."
            ),
            metadata={"region": "Northeast", "state": "Assam"},
        ),
        Document(
            page_content=(
                "Sikkim is India's second smallest state. It is known for "
                "Kanchenjunga, Buddhist monasteries, and its clean environment. "
                "Gangtok is the capital."
            ),
            metadata={"region": "Northeast", "state": "Sikkim"},
        ),
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="rag_docs",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    # Add documents to the vector store

    vector_store.add_documents(documents)

    print(f"Successfully added {len(documents)} documents to ChromaDB.")

    # get the first 5 documents from the vector store
    retrieved_docs = vector_store.get(limit=5) 

    # similarity search for a query
    response = vector_store.similarity_search(
        query="Which states are in western India?",
        k=2
    )

    print("Retrieved documents based on similarity search:")
    for doc in response:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)

    
    # search for a query with metadata filter
    response_with_filter = vector_store.similarity_search(
        query="Which states are in western India?",
        k=2,
        filter={"region": "West"}
    )

    print("Retrieved documents based on similarity search with filter:")
    for doc in response_with_filter:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)


if __name__ == "__main__":
    main()