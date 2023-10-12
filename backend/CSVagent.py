# 1. Setup
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load the API Key
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API Key not set in environment variables")

# Load CSV into a dataframe
file_path = "backend/agentDataset.csv"
df = pd.read_csv(file_path)
df_head = df.head().to_string()

# 2. Integrate CSV data with LLMChain
# Prepare a custom prompt using CSV data
prompt_template = '''
Your CSV data is as follows:
{df_head}

Given this data, how can I assist you?
'''

llm = ChatOpenAI(temperature=0.5, openai_api_key=api_key, model="gpt-4")
# Create an empty list for messages
messages_list = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add messages to the list
    messages_list.append(SystemMessage(content=prompt_template.format(df_head=df_head)))
    messages_list.append(HumanMessage(content=user_input))
    
    try:
        response = llm(messages=messages_list)
        print("Bot:", response.content)
    except Exception as e:
        print("An error occurred:", e)
        break

print("Goodbye!")
