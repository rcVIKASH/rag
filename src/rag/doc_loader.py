from langchain_community.document_loaders import TextLoader, PyPDFLoader


def main():
    text_loader = TextLoader("./src/data/notes.txt")
    text_docs = text_loader.load()

    pdf_loader = PyPDFLoader("./src/data/sample.pdf")
    pdf_docs = pdf_loader.load()

    print("-------------Text Loader-------------")
    print(text_docs[0])
    print("-------------PDF Loader page 1-------------")
    print(pdf_docs[0])
    print("-------------PDF Loader page 2-------------")
    print(pdf_docs[1])

    print("-----------PDF Document------------")
    print(pdf_docs)

    # lazy loading
    lazy_pdf_loader = PyPDFLoader("./src/data/sample.pdf")
    lazy_pdf_docs = lazy_pdf_loader.lazy_load()

    print("-------------Lazy PDF Loader-------------")
    for doc in lazy_pdf_docs:
        print(doc.metadata)


if __name__ == "__main__":
    main()
