from dotenv import load_dotenv

load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq


def main():

    # -----------------------------
    # Step 1: Fetch transcript
    # -----------------------------
    video_id = "wjZofJX0v4M"

    transcript = YouTubeTranscriptApi().fetch(video_id)

    # -----------------------------
    # Step 2: Convert transcript to Documents
    # -----------------------------
    documents = [
        Document(
            page_content=snippet.text,
            metadata={
                "video_id": video_id,
                "start": snippet.start,
                "duration": snippet.duration,
            },
        )
        for snippet in transcript
    ]

    # -----------------------------
    # Step 3: Split Documents
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )

    split_docs = text_splitter.split_documents(documents)

    # -----------------------------
    # Step 4: Embedding Model
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Step 5: Vector Store
    # -----------------------------
    vector_store = Chroma(
        collection_name="youtube_transcripts",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    # use ids in production to avoid duplicates, here we reset the collection for demonstration purposes
    # vector_store.reset_collection()

    vector_store.add_documents(split_docs)

    # -----------------------------
    # Step 6: Retriever
    # -----------------------------
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # -----------------------------
    # Step 7: Prompt
    # -----------------------------
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not present in the context, reply with "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    query = "Pm of india?"

    retrieved_docs = retriever.invoke(query)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.format(
        context=context_text,
        question=query,
    )

    # -----------------------------
    # Step 8: LLM
    # -----------------------------
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.4,
    )

    response = llm.invoke(final_prompt)

    print(response.content)


if __name__ == "__main__":
    main()
