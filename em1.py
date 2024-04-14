#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model_local=ChatOllama(model="mistral")

# 1. Splite data into chunks
mds=[
    r"Policy Analysis Using Advanced Methods.md",
    # "https://www.linkedin.com/in/michelle--yin/",
    # "https://www.air.org/resource/qa/meet-expert-michelle-yin",
    # "https://bpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/9/5450/files/2021/10/Michelle-Yin_Resume_2021_NU_oct-1.pdf", 在线pdf的识别效果不好
]

docs=[UnstructuredMarkdownLoader(md).load() for md in mds]
docs_list=[item for sublist in docs for item in sublist]
text_splitter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500,chunk_overlap=10)
doc_splits=text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
vectorstore=Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding= embeddings.OllamaEmbeddings(model='nomic-embed-text'), #原始代码写的是embedding= embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text') 但是持续报错
)
retriever=vectorstore.as_retriever()

# 3. Before RAG
print("Before RAG\n") #\n表示换行
before_rag_template="Who is {topic}"
before_rag_prompt= ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain= before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic":"Professor Michelle Yin"}))

# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template="""Answer the question based only on the following context:{context}
Question:{question}
"""
after_rag_prompt=ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain=(
    {"context":retriever,"question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("Who is Professor Michelle Yin?"))






