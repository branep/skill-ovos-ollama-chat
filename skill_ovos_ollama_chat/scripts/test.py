from ollama import Client
import re

look_ahead = ""


def chat():
    client = Client(host="http://ollama.lan")
    return client.chat(
        model="phi3",
        system="Your name is Jarvis and you are located in Bolton, Ontario. Your timezone is GMT-4. You always output numbers as words instead of numerals. Your answers really succinct",
        messages=[
            {
                "role": "system",
                "message": "Your name is Jarvis and you are located in Bolton, Ontario. Your timezone is GMT-4. You always output numbers as words instead of numerals. Your answers really succinct",
            },
            {
                "role": "user",
                "content": "What's your name?",
            },
        ],
        keep_alive=-1,
        stream=True,
        options = {
            "num_predict": 90,
        }
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
