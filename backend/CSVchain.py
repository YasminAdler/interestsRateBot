import os

api_key = "open_AI_key"


# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
# if api_key is None:
#     raise ValueError("API Key not set in environment variables")

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


file_path = "backend/agentDataset.csv"
df = pd.read_csv(file_path)
df_head = df.head().to_string()

prompt_template = '''
Your CSV data is as follows:
{df_head}
get as much details as you can from the user, initial investment, time of investment, preferred bank etc
Given this data, how can I assist you?
'''

llm = ChatOpenAI(temperature=0.5, openai_api_key=api_key, model="gpt-4")
messages_list = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages_list.append(SystemMessage(content=prompt_template.format(df_head=df_head)))
    messages_list.append(HumanMessage(content=user_input))
    
    try:
        response = llm(messages=messages_list)
        print("Bot:", response.content)
    except Exception as e:
        print("An error occurred:", e)
        break

print("להתראות!")
