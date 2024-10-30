from ollama import Client
import re

look_ahead = ""


def chat():
    client = Client(host="http://ollama.lan")
    return client.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "message": "Your name is Jarvis. You are located in Bolton, Ontario. Your timezone is GMT-4. Your answers are as short as possible. Always output numbers as words instead of numerals.",
            },
            {
                "role": "user",
                "content": "How do you change a light bulb?",
            },
        ],
        keep_alive=-1,
        stream=True,
    )


for chunk in chat():
    if look_ahead != "":
        token = chunk["message"]["content"]
        if "." in token and bool(re.search("[0-9]", look_ahead)):
            print(f"Found numbered list: {chunk}")
        # print(f"\"{chunk['message']['content']}\"", end="")
        print(chunk["message"]["content"], end="")
        # print(f'"{chunk}"')
        if chunk["done"]:
            print("End")
    look_ahead = chunk["message"]["content"]
