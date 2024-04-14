from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.messages import AIMessage, HumanMessage #tool to get AIMessage and HumanMessage
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor, Tool, tool
from langchain_openai import ChatOpenAI #Tool to chat with OpenAI.
from dotenv import load_dotenv
from tools.search_web import search_web
from tools.search_pdf import search_pdf
load_dotenv()



def consultant_agent(query_text):
    # Load the language model we use to control the agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [search_pdf, search_web]

    #Bind tool to the llm: so you can allow the llm to invoke the tools when needed
    llm_with_tools = llm.bind_tools(tools)

    #Store chat history
    chat_history = []

    #Create prompt to guide the agents, inject agent's memory into the prompt
    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a researcher who answers user questions by searching on the web, or searching on relevant pdf documents.
                Summarise your answer in 3 bullet points.  You must cite your sources.
                """,
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            (
                "system",
                """
                Remember to cite your sources using the url link
                """,
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"), #This will be passed via the chain, when the chain is processing openai tool message
        ]
    )


    #Define the agent
    agent = (
        {
            "input": lambda x: x["input"], #Input here is the first variable, inputted through user message
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"] #a component for formatting intermediate steps (agent action, tool output pairs) to input messages that can be sent to the model
            ),
            "chat_history": lambda x: x["chat_history"] #add another variable to the prompt to store chat history
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    #Define an agent executor:
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=2)

    #Invoke the agent
    result = agent_executor.invoke({"input": query_text, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=query_text),
            AIMessage(content=result["output"]),
        ]
    ) #Extend chat history with the messages from prior messages
    return(result)

if __name__ == "__main__":
    consultant_agent()

