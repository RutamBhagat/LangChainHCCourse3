### Create a chatbot
# Note: This has some memory persistance issues, It might be the way i have setup streamlit, It's definitely not an issue with langchain

from langchain.schema import AIMessage, HumanMessage
import streamlit as st
from streamlit_chat import message
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import requests
from pydantic import BaseModel, Field
import datetime
import wikipedia

# Note This is the same function from last lesson
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

# Note This is the same function from last lesson
# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'

@tool
def create_your_own(query: str) -> str:
    """This function can do whatever you would like once you fill it in """
    print(type(query))
    return query[::-1]

tools = [get_current_temperature, search_wikipedia, create_your_own]

class ChatBot:
    def __init__(self, tools):
        self.tools = tools
        self.functions = [convert_to_openai_function(f) for f in self.tools]
        self.model = ChatOpenAI(temperature=0).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=True, memory=self.memory)

    def get_response(self, query):
        result = self.qa.invoke({"input": query})
        return result['output']
    
def main():
    chatbot = ChatBot(tools)
    st.header("Conversational Agent Chatbot 🤖")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Prompt", placeholder="Enter your query here...", key="user_input")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if user_input:
        with st.spinner("Generating Response..."):
            response = chatbot.get_response(user_input)
            st.session_state["user_prompt_history"].append(user_input)
            st.session_state["chat_answers_history"].append(response)
            st.session_state["chat_history"].append((user_input, response))

    chat_history = st.session_state["chat_history"]
    print("Chatbot Memory: ", chat_history)

    if chat_history:
        st.subheader("Chat History")
        for user_prompt, chat_answer in zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
        ):
            message(user_prompt, is_user=True)
            message(chat_answer)

if __name__ == "__main__":
    main()
