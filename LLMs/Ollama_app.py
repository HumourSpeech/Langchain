import os
from dotenv import load_dotenv

# from langchain_community.llms import ollama
from langchain_ollama import OllamaLLM as ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
## ChatpromptTemplate works good with chat models
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Translator that translates sentences into {output_language}."),
        ("user", "{input_text}"),
    ]
)

## prompt template for non-chat models
from langchain_core.prompts import PromptTemplate
prompt1 = PromptTemplate(
    template="""
Translate the following text into {output_language}.
Return ONLY the translated text.

Text:
{input_text}
""",
    input_variables=["input_text", "output_language"],
)

## streamlit framework
st.title("Ollama LLM with Langchain")
input_text = st.text_input("Enter your text to translate:")
# input_language = st.text_input("Input Language (e.g., English):")
output_language = st.text_input("Output Language (e.g., Spanish):")

# Ollama gemma model(Text Completion model)
llm = ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt1 | llm | output_parser

# Using Chain operator
if st.button("Translate"):
    if input_text and output_language:
        result = chain.invoke({            
            # "input_language": input_language,
            "output_language": output_language,
            "input_text": input_text,
            }
        )
        st.write("Translated Text:", result)
    else:
        st.write("Please provide all inputs.")

# without using chain operator
# if st.button("Translate"):
#     if input_text and input_language and output_language:
#         formatted_prompt = prompt.format_messages(
#             question=input_text,
#             input_language=input_language,
#             output_language=output_language
#         )
#         response = llm.generate([formatted_prompt])
#         translated_text = output_parser.parse(response.generations[0][0].text)
#         st.write("Translated Text:", translated_text)
#     else:
#         st.write("Please provide all inputs.")

