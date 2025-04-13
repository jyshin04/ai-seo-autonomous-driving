import openai
import pandas as pd

data = pd.read_csv("part2_prompts.csv")

openai.api_key = API_KEY
messages = [ {"role": "system", "content": 
			"You are a intelligent assistant."} ] 

data.insert(2, column = "ChatGPT", value = None) 

row = 0
for question in data["Prompt"]: #for each question
    if isinstance(question, str): #checking if the row is not empty
        messages.append({"role": "user", "content": question}) 
        chat = openai.ChatCompletion.create(model="gpt-4o", messages=messages) 
        data.loc[row, "ChatGPT"] = chat.choices[0].message.content 
        messages.pop() 
    row += 1

data.to_csv("../data/raw/chatgpt_long_response.csv")

