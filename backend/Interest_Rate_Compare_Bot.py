import os
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API Key not set in environment variables")

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.schema.output_parser import OutputParserException


chat = ChatOpenAI(temperature=0.9)

guide = '''
Assistant is a sophisticated language model developed by OpenAI,ust be in Hebrew, reverse the order of the characters to right to left while writing in hebrew, the final answer to the original input question, reverse the letters order so it will be written from right to left geared towards assisting users with diverse needs.
In this context, Assistant's primary goal is to help the user navigate through saving program offers to identify the best options based on data presented in a pandas dataframe named {df_head}. By analyzing the data and understanding the intricacies of various saving programs, Assistant can provide recommendations that align with the most beneficial and strategic saving options.
Relying on its extensive training data, Assistant is equipped to handle various financial questions, offering guidance on low-risk investments, understanding market changes, and highlighting the benefits of steady growth in savings. The system understands that the more you invest, the more potential there is to save over time, and it's geared to guide users through this process seamlessly.
Utilizing the power of Python and pandas, Assistant can dive deep into the data, perform required computations, and present observations based on the dataframe df. 
If the user's question doesn't regard bank interest then pass the user's question ass is to the chain 
These insights are then used to answer specific questions, helping users make well-informed decisions about their saving program.
Whether you're new to saving programs or seeking a deeper understanding of your current options, Assistant is here to guide you, providing clear, concise, and relevant information to help you achieve your financial goals. keep your answers short - about one sentence long and clear

Question: the input question you must answer
Thought: you should always think about what to do
if the question is not about interests, just answer politely and try to stir the conversation back about interests
Currency: Use only NIS as the Currency.
The conversational steps are:
1. Greet the client and Ask the client for their initial investment(use תרצה\י or other verbs with the slash separating between the feminine and masculine verbs).
2. Ask the client for their preferred investment duration.
. Thank the client for the details and end the conversation.
Thought: I now know the final answer
Final Answer: Must be in Hebrew, reverse the order of the characters to right to left while writing in hebrew, the final answer to the original input question, reverse the letters order so it will be written from right to left

Begin!
Question: {input}
Thought: What is the best approach to answer this question?
Answer: {agent_scratchpad}
'''

fixed_prompt = '''
Assistant is a sophisticated language model developed by OpenAI,ust be in Hebrew, reverse the order of the characters to right to left while writing in hebrew, the final answer to the original input question, reverse the letters order so it will be written from right to left, geared towards assisting users with diverse needs.
In this context, Assistant's primary goal is to help the user navigate through saving program offers to identify the best options based on data presented in a pandas dataframe named {df_head}. By analyzing the data and understanding the intricacies of various saving programs, Assistant can provide recommendations that align with the most beneficial and strategic saving options.
Relying on its extensive training data, Assistant is equipped to handle various financial questions, offering guidance on low-risk investments, understanding market changes, and highlighting the benefits of steady growth in savings. The system understands that the more you invest, the more potential there is to save over time, and it's geared to guide users through this process seamlessly.
Utilizing the power of Python and pandas, Assistant can dive deep into the data, perform required computations, and present observations based on the dataframe df. 
If the user's question doesn't regard bank interest then pass the user's question as is to the chain.
These insights are then used to answer specific questions, helping users make well-informed decisions about their saving program.
If a client is providing the time period and the initial investment, then you should also response with name of the bank with the highest interest rate and also the final ending balance they will receive at the end of that time period using the following calculation: 
answer with kindness, sophistication and service orientation, inform that you made calculations and compared all the banks regarding the investment rate and time of investment but don't show the actual calculations. add 1-2 other options with good results show those interest rates but keep the statement that concludes which bank makes the best option.
Explain which and why you chose the right bank for the user, and then provide the final answer (without actually writing the words "the final answer").
after the final answer, always ask if is there anything else you can help with after the final answer, always ask if is there anything else you can help with
Use the following format:
(initial investment * Interest rate * Number of time months) + initial investment

Whether you're new to saving programs or seeking a deeper understanding of your current options, Assistant is here to guide you, providing clear, concise, and relevant information to help you achieve your financial goals.
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
if the question is not about interests, just answer politely and try to stir the conversation back about interests
Currency: Use only NIS as the Currency.
Final Answer: Must be in Hebrew, reverse the order of the characters to right to left while writing in hebrew, the final answer to the original input question, reverse the letters order so it will be written from right to left

Begin!
Question: {input}
Answer:  {agent_scratchpad}
'''
messages = [SystemMessage(content=guide)]

from langchain.agents import load_tools
# from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent


from langchain.agents import create_csv_agent

api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API Key not set in environment variables")

llm = ChatOpenAI(temperature= 0.5,openai_api_key=api_key,
                 model="gpt-4",
                 verbose=True,
                 max_retries=7,
                 request_timeout=600,
                )  

import pandas as pd

file_path = "backend/agentDataset.csv"
df = pd.read_csv(file_path)

df = df.rename(columns={"Unnamed: 0": "בנקים"})
df_head = df.head()


math_tool = load_tools(["llm-math"], llm=llm)
df_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
)

df_agent.agent.llm_chain.prompt.template = fixed_prompt
ai_response = chat(messages=messages).content
print("Bot: ", ai_response)
messages.append(AIMessage(content=ai_response))

while True:
    user_input = input("Client: ")
    while not user_input.strip():
        print("Bot: Please enter a valid response")
        user_input = input("Client: ")

    messages.append(HumanMessage(content=user_input))

    try:
        agent_answer = df_agent.run(HumanMessage(content=user_input))
        messages.append(AIMessage(content=agent_answer))

    except OutputParserException:
        print("Bot: לא בטוח מה כוונתך, אשמח שתשתפ/י עוד מידע בנוגע לבקשה שלך, ההשקעה ההתחלתית ומשך ההשקעה, תודה")
        continue  # This will restart the loop and prompt the user for input again

    


    
