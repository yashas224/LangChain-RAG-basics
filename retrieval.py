import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

print('initialization process')
llm  = ChatOpenAI(model="gpt-5")
embeddings = OpenAIEmbeddings(
model="text-embedding-ada-002")
index=os.getenv("VECTOR_STORE_INDEX_NAME")
vector_store = PineconeVectorStore(index_name=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

StringOutputParser

prompt = """
You are a helpful AI assistant.

Use only the information provided in the context to answer the question.
If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:

"""
prompt = ChatPromptTemplate.from_template(prompt)

def retrieveWithoutLangChainExpressionLanguage(query):
    print("retrieveWithoutLangChainExpressionLanguage")
    #  retrieve
    revelant_doc : list[Document] = retriever.invoke(query)

    print(f"revelant_doc {revelant_doc}")
    string_doc_list = [doc.page_content for doc in revelant_doc]
    context = "\n\n".join(string_doc_list)


    #  Augment

    message = prompt.invoke({
        'context':context,'question':query
    })

    #  Generate

    response= llm.invoke(message)

    print(f"revelant_doc {revelant_doc}")
    print("--"* 60)
    print("Response")


    print(response.content)


def retrieveWithLangChainExpressionLanguage(query):

    pass


if(__name__ == '__main__'):
    query = 'What is pinecone in machine learning ??'
    retrieveWithoutLangChainExpressionLanguage(query)


