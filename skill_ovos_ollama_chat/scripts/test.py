from ollama import Client


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
                "content": "As briefly as possible, what is the tallest building in the world",
            },
        ],
        keep_alive=-1,
        stream=True,
    )


for chunk in chat():
    # print(f"\"{chunk['message']['content']}\"", end="")
    print(chunk["message"]["content"], end="")
    # print(f'"{chunk}"')
    if chunk["done"]:
        print("End")
