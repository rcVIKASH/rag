from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Retrieve is a function that demonstrates how to use the Chroma vector store with HuggingFace embeddings to retrieve documents based on a query.
# Retrieve is runnable.


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
                "Kerala is known as God's Own Country for its backwaters, beaches, "
                "and lush greenery. Thiruvananthapuram is the capital. The state "
                "has one of the highest literacy rates in India."
            ),
            metadata={"region": "South", "state": "Kerala"},
        ),
        Document(
            page_content=(
                "Karnataka is home to Bengaluru, India's technology hub. "
                "The state is known for Mysore Palace, Hampi, coffee plantations, "
                "and a thriving IT industry."
            ),
            metadata={"region": "South", "state": "Karnataka"},
        ),
        Document(
            page_content=(
                "Andhra Pradesh is famous for Tirupati Temple, rich agriculture, "
                "and a long coastline along the Bay of Bengal. Amaravati is the "
                "legislative capital of the state."
            ),
            metadata={"region": "South", "state": "Andhra Pradesh"},
        ),
        Document(
            page_content=(
                "Telangana was formed in 2014. Hyderabad is its capital and is "
                "known for Charminar, biryani, and a rapidly growing IT sector."
            ),
            metadata={"region": "South", "state": "Telangana"},
        ),
        Document(
            page_content=(
                "Maharashtra is India's second-most populous state. Mumbai, the "
                "financial capital of India, is its capital. The state is famous "
                "for Bollywood, Ajanta and Ellora caves, and automobile industries."
            ),
            metadata={"region": "West", "state": "Maharashtra"},
        ),
        Document(
            page_content=(
                "Goa is India's smallest state by area. Panaji is the capital. "
                "The state is famous for beaches, Portuguese architecture, seafood, "
                "and tourism."
            ),
            metadata={"region": "West", "state": "Goa"},
        ),
        Document(
            page_content=(
                "Madhya Pradesh is known as the Heart of India due to its central "
                "location. Bhopal is the capital. The state is famous for Khajuraho "
                "Temples, Kanha National Park, and tiger reserves."
            ),
            metadata={"region": "Central", "state": "Madhya Pradesh"},
        ),
        Document(
            page_content=(
                "Chhattisgarh is rich in mineral resources and forests. Raipur is "
                "the capital. The state is known for steel production, waterfalls, "
                "and tribal culture."
            ),
            metadata={"region": "Central", "state": "Chhattisgarh"},
        ),
        Document(
            page_content=(
                "Uttar Pradesh is India's most populous state. Lucknow is the "
                "capital. The state is home to the Taj Mahal in Agra, Varanasi, "
                "and Prayagraj."
            ),
            metadata={"region": "North", "state": "Uttar Pradesh"},
        ),
        Document(
            page_content=(
                "Uttarakhand is known as the Land of Gods because of pilgrimage "
                "sites like Kedarnath and Badrinath. Dehradun is the capital. "
                "The state is famous for the Himalayas and adventure tourism."
            ),
            metadata={"region": "North", "state": "Uttarakhand"},
        ),
        Document(
            page_content=(
                "Bihar is one of India's oldest inhabited regions. Patna is the "
                "capital. The state is famous for Nalanda University, Bodh Gaya, "
                "and the fertile Gangetic plains."
            ),
            metadata={"region": "East", "state": "Bihar"},
        ),
        Document(
            page_content=(
                "Jharkhand is rich in coal and iron ore reserves. Ranchi is the "
                "capital. The state is known for waterfalls, forests, and mining."
            ),
            metadata={"region": "East", "state": "Jharkhand"},
        ),
        Document(
            page_content=(
                "West Bengal is known for Kolkata, the Sundarbans mangrove forest, "
                "and Durga Puja celebrations. Kolkata is the capital."
            ),
            metadata={"region": "East", "state": "West Bengal"},
        ),
        Document(
            page_content=(
                "Odisha is famous for the Jagannath Temple in Puri, Konark Sun "
                "Temple, and Chilika Lake. Bhubaneswar is the capital."
            ),
            metadata={"region": "East", "state": "Odisha"},
        ),
        Document(
            page_content=(
                "Assam is known for tea gardens, Kaziranga National Park, and the "
                "Brahmaputra River. Dispur is the capital."
            ),
            metadata={"region": "Northeast", "state": "Assam"},
        ),
        Document(
            page_content=(
                "Sikkim is India's least populous state. Gangtok is the capital. "
                "The state is famous for Kanchenjunga, Buddhist monasteries, and "
                "organic farming."
            ),
            metadata={"region": "Northeast", "state": "Sikkim"},
        ),
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    query = "Which state is famous for tiger reserves?"

    results = retriever.invoke(query)
    print(results)

    # RAG, Maximal Marginal Relevance (MMR)

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.5}
    )

    query = "Which state is famous for tiger reserves?"

    results = retriever.invoke(query)
    print(results)


if __name__ == "__main__":
    main()
