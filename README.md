# LangChain Practice Repository

Welcome to this comprehensive repository dedicated to mastering **LangChain** and **Generative AI** concepts. This project serves as a structured learning path, documenting various techniques from basic data ingestion to building complex RAG (Retrieval-Augmented Generation) applications and deploying them as APIs.

Here you will find well-documented code examples and notebooks covering the entire lifecycle of an LLM-powered application.

## 📂 Project Structure

The repository is organized into logical modules, each focusing on a specific aspect of the LangChain framework:

- **DataIngestion/**: Techniques for loading data from various sources.
- **DataTransformation/**: Methods for splitting and chunking text/data.
- **Embeddings/**: Converting text into vector representations using different providers.
- **VectorDatabase/**: Storing and retrieving vector embeddings.
- **LLMs/**: Working with Large Language Models, Chains, Chatbots, and API deployment.
- **Projects/**: End-to-end applications and complex use cases.

---

## 🚀 Concepts & Techniques Covered

### 1. Data Ingestion
The first step in any RAG pipeline is getting data into the system. This module demonstrates how to load documents from diverse sources:
- **TextLoader**: Loading standard `.txt` files.
- **PyPDFLoader**: Extracting text from PDF documents.
- **WebBaseLoader**: Scraping content from websites. Includes advanced techniques like using `bs4` (BeautifulSoup) to filter specific HTML classes (e.g., extracting only the main content).
- **ArxivLoader**: Fetching research papers directly from Arxiv.
- **WikipediaLoader**: Querying and downloading content from Wikipedia.

### 2. Data Transformation (Text Splitting)
Once data is loaded, it often needs to be split into smaller, manageable chunks for processing. We explore several splitting strategies:
- **RecursiveCharacterTextSplitter**: The most common splitter, recursively splitting by characters to keep related text together.
- **CharacterTextSplitter**: Simple splitting based on a specific character separator.
- **HTMLHeaderTextSplitter**: Splitting HTML content based on header tags (`<h1>`, `<h2>`, etc.) to preserve structural context.
- **RecursiveJsonSplitter**: Specialized splitter for handling JSON data structures.

### 3. Embeddings
Embeddings convert text into numerical vectors that capture semantic meaning. This repository covers multiple embedding models:
- **OpenAIEmbeddings**: Using OpenAI's powerful embedding models (e.g., `text-embedding-3-large`).
- **HuggingFaceEmbeddings**: Utilizing open-source models from Hugging Face (e.g., `all-MiniLM-L6-v2`) for local or cost-effective embeddings.
- **OllamaEmbeddings**: Running embeddings locally using Ollama (e.g., `gemma:2b`).

### 4. Vector Databases & Retrievers
To perform efficient similarity searches, we store embeddings in vector databases. We demonstrate:
- **Chroma**: An AI-native open-source vector database.
- **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors.
- **Retrievers**: Understanding `VectorStoreRetriever`, similarity search, and Maximum Marginal Relevance (MMR).

### 5. LLMs, Chains & Chatbots
This is the core where we interact with Language Models and build workflows:
- **Models**:
  - **ChatOpenAI**: Interacting with OpenAI's GPT models.
  - **ChatOllama / OllamaLLM**: Running open-source models locally (e.g., Gemma, Llama).
  - **ChatGroq**: Using Groq's high-speed inference API (e.g., `openai/gpt-oss-120b`).
- **Prompts**:
  - **PromptTemplate** & **ChatPromptTemplate**: Structuring inputs for LLMs.
  - **MessagesPlaceholder**: Handling dynamic history in prompts.
- **Chains**:
  - **LCEL (LangChain Expression Language)**: Building chains using the `|` syntax.
  - **Retrieval Chains**: Combining LLMs with Vector Stores to answer questions based on documents.
- **Chatbots**:
  - **RunnableWithMessageHistory**: Managing session history for stateful conversations.
  - **Memory**: Handling chat history manually and automatically.

### 6. Deployment (LangServe)
- **LangServe**: Deploying LangChain runnables and chains as REST APIs using **FastAPI**.

---

## 🛠️ Projects & Applications

### Conversational QA Chatbot (`Projects/convoQAchatbot.ipynb`)
A sophisticated RAG-based chatbot that:
1.  Scrapes data from the web (e.g., Analytics Vidhya).
2.  Maintains **Chat History** to understand context across multiple turns.
3.  Uses a **History Aware Retriever** to reformulate queries based on previous interactions.
4.  Answers questions using retrieved context.

### LangServe API (`LLMs/LangserveAPI.py`)
A deployment example showing how to:
1.  Create a translation chain using `ChatGroq`.
2.  Wrap the chain in a **FastAPI** application using `add_routes`.
3.  Serve the application locally for external access.

### Simple RAG Application (`LLMs/simpleApp.ipynb`)
A complete end-to-end pipeline demonstrating:
1.  Loading data from the web.
2.  Splitting text into chunks.
3.  Creating embeddings.
4.  Storing in a Vector DB (FAISS/Chroma).
5.  Retrieving relevant context.
6.  Generating answers using an LLM.

### Streamlit App (`LLMs/Ollama_app.py`)
A interactive web application built with **Streamlit** that utilizes **Ollama** models for translation and other tasks, showcasing how to deploy LangChain logic in a user-friendly interface.

---

## 🔮 Future Updates
This repository is a living document of my learning journey. Future updates will include:
- Advanced RAG techniques (Multi-query, RAG-Fusion).
- Agents and Tool usage.
- LangGraph for building stateful, multi-actor applications.
- More complex deployment examples.

Stay tuned!