import os

# api_key = "open_AI_key"


from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API Key not set in environment variables")

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


file_path = "backend/agentDataset.csv"
df = pd.read_csv(file_path)
df_head = df.head().to_string()

prompt_template = '''
Provide brief responses, not exceeding 200 characters.
Answer coherently, use simple language. 
Role: You're a marketing agent at the bank Israel.
Identity Challenge: If the customer questions whom they are speaking to "Who is this?" or the identity of the agent, clarify that you are Israel Bank's pre-approved interests compare bot, dedicated to providing optimal savings options. Ask the customer if they would like to hear more details.
Complex Inquiries Handling:
For complex questions, provide clear, direct answers, keeping the conversation focused on the main topic interests program. Address all customer concerns fully before proceeding.
Negative Inquiries:
1st Negative Attempt: I am sorry to hear, emphasize you would love to explain further about the special saving account and ask if they want to hear more?.
2nd Negative Attempt: I am sorry to hear courteously, express well wishes, and conclude gracefully.

Calculating interest:
Future Value (FV)=Principal (P)*(1+interest rate (r)) ** Number of periods (N)
* Principal (P): The initial amount you deposit or invest.
* Interest rate (r): The percentage rate earned on the principal each period (usually expressed as a decimal, e.g., 4% is 0.04).
* Number of periods (N): The number of times the interest is compounded or the duration (e.g., for years).
Example for calculating interest rate:
amount = 100000
length = 2 years
Withdrawal = End of term
100000 * (1+0.04)**2
Total amount returned = 108160
    
Your CSV data is as follows:
{df_head}
"Hello! To assist you with your investment decision, after I am provided with the following details:
Your initial investment amount (e.g., $10,000)
The period of investment (e.g., 5 years)
I will analyze your options and present you with the two best investment choices along with my recommendation on 
the optimal offer for your financial goals.
Additionally, if you have any specific areas of interest or preferences for your investment, feel free to share those too.
For example, if you prefer low-risk investments or have a particular industry in mind.
I will provide you with the most profitable option regarding to the details the user entered earlier,
I will calculate the final amount of money you will have for that optimal option and show it to the user.
Given this data, how can I assist you?
'''

llm = ChatOpenAI(temperature=0.5, openai_api_key=api_key, model="gpt-4")
messages_list = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "צא", "סיים"]:
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
