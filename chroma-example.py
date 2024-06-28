from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


if __name__ == "__main__":
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
            metadata={"source": "fish-pets-doc"},
        ),
        Document(
            page_content="Parrots are intelligent birds capable of mimicking human speech.",
            metadata={"source": "bird-pets-doc"},
        ),
        Document(
            page_content="Rabbits are social animals that need plenty of space to hop around.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
    )

    result = vectorstore.similarity_search("cat")
    print(result)
    print("=" * 10)

    result = vectorstore.similarity_search_with_score("cat")
    print(result)
    print("=" * 10)

    embedding = OpenAIEmbeddings().embed_query("cat")
    result = vectorstore.similarity_search_by_vector(embedding)
    print(result)
