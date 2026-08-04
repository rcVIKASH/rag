
from dotenv import load_dotenv
load_dotenv()
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import ( HuggingFaceEmbeddings )
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document

def main():
   

   documents = [
    Document(
        page_content=(
            "Python is a high-level programming language known for its simple "
            "syntax and readability. It is widely used for web development, "
            "automation, scripting, and data science."
        ),
        metadata={"category": "Programming", "topic": "Python"},
    ),

    Document(
        page_content=(
            "Pandas is a Python library designed for data manipulation and "
            "analysis. It provides DataFrame and Series data structures for "
            "working with tabular datasets."
        ),
        metadata={"category": "Library", "topic": "Pandas"},
    ),

    Document(
        page_content=(
            "NumPy is the fundamental package for numerical computing in Python. "
            "It offers efficient multidimensional arrays and mathematical "
            "operations."
        ),
        metadata={"category": "Library", "topic": "NumPy"},
    ),

    Document(
        page_content=(
            "Scikit-learn is a popular machine learning library in Python. "
            "It includes algorithms for classification, regression, clustering, "
            "and model evaluation."
        ),
        metadata={"category": "Machine Learning", "topic": "Scikit-learn"},
    ),

    Document(
        page_content=(
            "Flask is a lightweight Python web framework. It is commonly used "
            "for building REST APIs, web applications, and backend services."
        ),
        metadata={"category": "Web", "topic": "Flask"},
    ),

    Document(
        page_content=(
            "FastAPI is a modern Python framework for building high-performance "
            "REST APIs. It provides automatic API documentation and uses type "
            "hints for validation."
        ),
        metadata={"category": "Web", "topic": "FastAPI"},
    ),

    Document(
        page_content=(
            "Django is a full-stack Python web framework that includes an ORM, "
            "authentication system, admin interface, and many built-in features "
            "for rapid application development."
        ),
        metadata={"category": "Web", "topic": "Django"},
    ),

    Document(
        page_content=(
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn patterns from data without being explicitly "
            "programmed."
        ),
        metadata={"category": "AI", "topic": "Machine Learning"},
    ),

    Document(
        page_content=(
            "Deep learning is a subset of machine learning that uses neural "
            "networks with multiple layers. It is widely used for computer "
            "vision, speech recognition, and natural language processing."
        ),
        metadata={"category": "AI", "topic": "Deep Learning"},
    ),

    Document(
        page_content=(
            "Natural Language Processing (NLP) focuses on enabling computers to "
            "understand, generate, and analyze human language. Large language "
            "models are a major application of NLP."
        ),
        metadata={"category": "AI", "topic": "NLP"},
    ),

    Document(
        page_content=(
            "LangChain is a framework for building applications powered by large "
            "language models. It provides components for prompts, chains, "
            "retrievers, memory, and agents."
        ),
        metadata={"category": "LLM", "topic": "LangChain"},
    ),

    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with language models. Relevant documents are retrieved from a "
            "knowledge base before generating an answer."
        ),
        metadata={"category": "LLM", "topic": "RAG"},
    ),

    Document(
        page_content=(
            "A vector database stores text embeddings for semantic search. "
            "Popular vector stores include Chroma, FAISS, Pinecone, and Weaviate."
        ),
        metadata={"category": "Database", "topic": "Vector Store"},
    ),

    Document(
        page_content=(
            "Embeddings convert text into dense numerical vectors. Similar "
            "meaning results in vectors that are close together in embedding "
            "space, enabling semantic search."
        ),
        metadata={"category": "LLM", "topic": "Embeddings"},
    ),

    Document(
        page_content=(
            "Prompt engineering is the practice of designing effective prompts "
            "to improve the quality, accuracy, and consistency of responses from "
            "large language models."
        ),
        metadata={"category": "LLM", "topic": "Prompt Engineering"},
    ),
    ]

   # Initialize the embeddings model
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   vectorstore = Chroma.from_documents(documents, embeddings)

   similarity_retriever = vectorstore.as_retriever(
       search_type="similarity",
       search_kwargs={"k": 2}
   )

   llm = ChatGroq(
       model="openai/gpt-oss-120b",
       temperature=0.7,
   )

   mqr_retriever = MultiQueryRetriever.from_llm(
       retriever=similarity_retriever,
       llm=llm,
   )

   query="How do I build applications?"

   similarity_results = similarity_retriever.invoke(query)
   mqr_results = mqr_retriever.invoke(query)

   print("Similarity Results:")
   for result in similarity_results:
       print(f" - {result}")

   print("\nMulti-Query Results:")
   for result in mqr_results:
       print(f" - {result}")



if __name__ == "__main__":
    main()  
