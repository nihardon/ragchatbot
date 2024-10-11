from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


loader = WebBaseLoader("https://www.foxsports.com/nba/lebron-james-player-game-log")
data = loader.load()

vectorstore = Chroma.from_documents(documents=data,
                                    embedding=OllamaEmbeddings(model="nomic-embed-text"))

retriever = vectorstore.as_retriever(search_kwargs={"k" : 1})

template = """Answer the following question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = Ollama(model="mistral")

rag_chain = (
    {"context" : retriever, "question" : RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke({"How many points did Lebron James score on 4/29?"})
print(response)