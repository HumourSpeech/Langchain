from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

## Creating Prompt Template
system_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', "{text}")
])

parser = StrOutputParser()

## Creating Chain
chain = prompt | model | parser

## APP definition
app = FastAPI(title = "Langchain Groq API",
                 description="An API for Langchain using Groq LLM and langserve",
                 version="0.1")

## Adding chain route to the app
add_routes(
    app,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)