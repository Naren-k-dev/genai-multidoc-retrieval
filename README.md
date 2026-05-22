### Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM

To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT

Traditional question-answering systems are limited when handling multiple research documents because they lack efficient retrieval and contextual reasoning capabilities. Large Language Models (LLMs) alone cannot effectively manage and retrieve relevant information from multiple external research papers without a retrieval mechanism.

The objective of this experiment is to develop a multidocument retrieval agent using LlamaIndex and function-calling agents. The system should be capable of loading multiple research papers, creating vector indexes for semantic retrieval, dynamically selecting appropriate tools for different documents, and generating accurate responses for user queries by synthesizing information from multiple research articles.

### DESIGN STEPS
### STEP 1:

Download and load multiple research papers from OpenReview and create query tools for each document using LlamaIndex.

###  STEP 2:

Create vector indexes and object retrievers for semantic retrieval and tool selection across multiple research papers.

### STEP 3:

Implement a FunctionCallingAgentWorker using LlamaIndex to dynamically retrieve relevant tools and generate contextual answers for user queries.

PROGRAM
```
import os
import openai
import nest_asyncio
import urllib.request

from dotenv import load_dotenv, find_dotenv

nest_asyncio.apply()


# Load API Key
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']


# Research Paper URLs

urls = [
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=yV6fD7LYkF"
]


# PDF File Names

papers = [
    "loftq.pdf",
    "swebench.pdf",
    "values.pdf"
]


# Download Papers

for url, paper in zip(urls, papers):

    print(f"Downloading {paper}...")

    urllib.request.urlretrieve(url, paper)


# Import Required Libraries

from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader
from llama_index.core.objects import ObjectIndex

from llama_index.llms.openai import OpenAI

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner


# Create Tools for Each Paper

paper_to_tools_dict = {}

for paper in papers:

    print(f"Getting tools for paper: {paper}")

    documents = SimpleDirectoryReader(
        input_files=[paper]
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=f"{Path(paper).stem}_summary",
        description=f"Provides summary of {paper}"
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=f"{Path(paper).stem}_vector",
        description=f"Answers detailed questions from {paper}"
    )

    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]


# Combine All Tools

all_tools = [
    t for paper in papers
    for t in paper_to_tools_dict[paper]
]


# Create Object Index

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)


# Create Tool Retriever

obj_retriever = obj_index.as_retriever(
    similarity_top_k=3
)


# Create LLM

llm = OpenAI(
    model="gpt-3.5-turbo"
)


# Create Function Calling Agent

agent_worker = FunctionCallingAgentWorker.from_tools(

    tool_retriever=obj_retriever,

    llm=llm,

    system_prompt="""
You are an agent designed to answer queries
over multiple research papers.

Always use the provided tools
to answer user questions.

Do not rely on prior knowledge.
""",

    verbose=True
)


# Create Agent Runner

agent = AgentRunner(agent_worker)


# Query 1

 
response = agent.query(
    "Compare SWEBench and LoftQ papers"
)
print("Narendran K - 212223230135")
print(response)


# Query 2

response = agent.query(
    "Compare SWEBench and LoftQ papers"
)

print(response)


# Query 3

response = agent.query(
    "Which paper discusses evaluation methods?"
)

print(response)
```

### OUTPUT

![alt text](<Screenshot 2026-05-22 124628.png>)

### RESULT

Thus, the multidocument retrieval agent was successfully designed and implemented using LlamaIndex, vector indexing, object retrieval, and function-calling agents. The system successfully retrieved and synthesized information from multiple research papers and generated relevant, concise, and contextual responses for diverse user queries.