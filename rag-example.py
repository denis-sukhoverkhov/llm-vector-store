from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


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
        Document(
            page_content="A lynx is any of the four extant species within the medium-sized wild cat genus Lynx.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    response = rag_chain.invoke("tell me about lynx")
    print(response.content)
