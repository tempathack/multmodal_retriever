import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from configs import configs
import chromadb
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from typing import List,Annotated
import numpy as np
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from dotenv import load_dotenv
from collections import defaultdict
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from utils.utils import *
preprocess=False
load_dotenv()

# Sanitize your chunks before calling the API
if preprocess:
    # Load documents from directory
    loader = DirectoryLoader('./preprocessed_data/text_files', glob="*.txt")
    documents = loader.load()

    # Split documents into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0) # make sure each document is iin a split
    chunk_sources = []
    chunks = []
    ct=0
    for doc in documents:

        for chunk in text_splitter.split_text(doc.page_content):
            chunks.append(sanitize_string(chunk))
            chunk_sources.append(doc.metadata['source'])


    chunks = sanitize_input(chunks)

    df=pd.DataFrame(data={'chunk':chunks,'chunk_src':chunk_sources})


class VectorStoreRetriever:
    def __init__(self, vectors, oai_client, df: pd.DataFrame):
        self._arr = vectors
        self._client = oai_client
        self.df = df

    @classmethod
    def from_docs(cls, df):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Use PersistentClient instead of EphemeralClient
        persistent_client = chromadb.PersistentClient(path="./dbs")

        openai_lc_client = Chroma.from_texts(
            df.chunk.tolist(),
            embeddings,
            client=persistent_client,
            collection_name="openai_collection"
        )

        vecs = openai_lc_client._collection.get(include=['embeddings', 'documents'])
        return cls(vecs, openai_lc_client, df)

    @classmethod
    def load(cls):
        persistent_client = chromadb.PersistentClient(path="./dbs")
        collection = persistent_client.get_collection("openai_collection")

        vecs = collection.get(include=['embeddings', 'documents'])

        # Reconstruct df from saved data
        df = pd.DataFrame({
            'chunk': vecs['documents'],
            'chunk_src': [vecs['documents'][i] for i in range(len(vecs['documents']))]  # You might want to adjust this
        })

        openai_lc_client = Chroma(
            client=persistent_client,
            collection_name="openai_collection",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
        )

        return cls(vecs, openai_lc_client, df)

    def query(self, query: str, k: int = 5) -> pd.DataFrame:
        store = defaultdict(list)
        results_with_scores = self._client.similarity_search_with_score(query, k=k)

        for doc, score in results_with_scores:
            doc_index = np.argwhere(self.df.chunk.values.reshape(-1) == doc.page_content)[0]
            store['files'].append(self.df.iloc[doc_index].chunk_src.squeeze())
            store['scores'].append(score)
            store['content'].append(doc.page_content)
            store['embeddings'].append(
                self._arr['embeddings'][np.argwhere(np.array(self._arr['documents']) == doc.page_content)[0][0]]
            )

        return pd.DataFrame(store)


def lookup_policy(query: Annotated[
    str, 'query to retrieve data from the database that should help you answering the question']) -> Annotated[
    str, 'String corpus with similarity search data from a Rag that should utilize you with information to answer the question it is split into the documents name and the documents content']:
    """Use this before you answer any question. This contains all information that you need and can be your single point of truth."""
    df: pd.DataFrame = retriever.query(query, k=5)

    string = 'The following information seems to have answers based on the query asked:\n'
    for file, content in zip(df['files'].tolist(), df['content'].tolist()):
        string += f'Document: {file} with Content: {content}\n'

    return string


# Usage
# To create and save embeddings:
# df = pd.read_csv('your_data.csv')
#retriever = VectorStoreRetriever.from_docs(df)

# To load saved embeddings:
retriever = VectorStoreRetriever.load()


@tool
def lookup_policy(query: Annotated[str,'comprehensive elaborated query to retrieve data from the database that should help you answering the question']) -> Annotated[str,'String corpus with similarity search data from a Rag that should utilize you with information to answer the question it is splittet into teh documents name and the documents content']:
    """use this before you answer any question this contains all information that you need and can be your single point of truth """
    df :pd.DataFrame = retriever.query(query, k=20)

    string='The following information seem to have answers based on the query asked'
    for file,content in zip(df['files'].tolist(),df['content'].tolist()):
        string+= f'Document:{file} with Content:{content}'+'\n'


    return string


class Response(BaseModel):
    """Final response to the question being asked"""
    answer: str = Field(description = "The final answer to respond to the user")
    sources: List[str] = Field(description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information")
    document_names:List[str]=Field(description='list of all documents that you used to retrieve the information')

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """### Expert Prompt for File Share Specialist ###

As an esteemed file share expert, proficient in utilizing advanced tools, your role is to address user inquiries with the depth of knowledge derived from the vast array of documents stored on the file share system. Equipped with a comprehensive repository of documents related to the municipal building department, you are tasked with expertly responding to all questions within this domain.

Your access to a trove of documents allows you to provide well-informed answers, leveraging your expertise to guide users in navigating the intricacies of the topic. If there happens to be a query that falls beyond the scope of your expertise, it is essential to communicate this clearly without fabricating responses. Support your responses with cited references to ensure accuracy and credibility in addressing user queries. 

Remember, your proficiency lies in understanding and interpreting the documents at hand to offer precise and relevant insights to users seeking guidance on matters concerning the municipal building department. Your responses should reflect your expertise and be grounded in verifiable information contained within the shared documents."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(temperature=0,   model="gpt-4o",max_tokens=None,streaming=True)
llm_with_tools = llm.bind_functions([lookup_policy, Response])
agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)


@st.cache_resource()
def get_agent():
    return  AgentExecutor(tools=[lookup_policy], agent=agent, verbose=True)


if __name__=='__main__':
    get_agent()