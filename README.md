# Development of a PDF-Based Question-Answering Chatbot Using LangChain

## AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

## PROBLEM STATEMENT:
Searching through large PDFs manually is slow and inefficient. Traditional keyword searching fails to understand the actual meaning of questions. This project solves this by creating a chatbot pipeline that reads a PDF, saves its meaning in a database, and uses an AI model to answer user questions instantly and accurately.

## DESIGN STEPS:

### STEP 1:
**Load and Split PDF Documents:** Use `PyPDFLoader` to read all the text pages from the PDF file. Then, use `RecursiveCharacterTextSplitter` to cut the long text into small, clean chunks so the AI model can read it easily.

### STEP 2:
**Create Vector Embeddings and Database:** Pass the text chunks into `OpenAIEmbeddings` to convert the words into math vectors. Save these vectors into a `Chroma` database so the chatbot can perform fast semantic searches.

### STEP 3:
**Setup RetrievalQA Chain:** Initialize the large language model (`ChatOpenAI`) and link it to the vector database using LangChain's `RetrievalQA` module. When a user asks a question, the system pulls the correct matching text and uses the LLM to generate a natural answer.

### PROGRAM:

```
%pip install -q langchain langchain-community langchain-openai langchain-chroma langchain-text-splitters pypdf python-dotenv


import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found! Check your .env file.")

print("API key loaded successfully!")


pdf_path = "document.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError("Please place document.pdf in the notebook folder.")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("PDF split successfully!")
print("Total chunks:", len(chunks))


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Embeddings created successfully!")
print("Documents stored in ChromaDB!")


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created successfully!")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("AI model connected successfully!")


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the question using only the following context.

    If the answer is not available in the context,
    say "The answer is not available in the PDF."

    Context:
    {context}
    """),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda question: question
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("PDF Question-Answering Chatbot is ready!")

while True:

    question = input("\nAsk a question about your PDF (type exit to stop): ")

    if question.lower().strip() == "exit":
        print("Chatbot closed!")
        break

    if not question.strip():
        print("Please enter a valid question.")
        continue

    answer = qa_chain.invoke(question)

    print("\nChatbot Answer:")
    print(answer)

    print("-" * 50)

 ```


### OUTPUT:
<img width="1338" height="530" alt="Screenshot 2026-11-04 091802" src="https://github.com/user-attachments/assets/97f0be1a-848c-4c41-8c55-9355ee15617c" />



## RESULT:
The PDF-based question-answering chatbot was designed and implemented successfully using LangChain. The program accurately extracted text from the document, generated vector indices, and answered content-specific user queries correctly without any errors.
