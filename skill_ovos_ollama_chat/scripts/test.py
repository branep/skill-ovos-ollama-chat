from ollama import Client


def chat():
    client = Client(host="http://ollama.lan")
    return client.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": "What is the biggest mammal on earth?",
            },
        ],
        keep_alive=-1,
        stream=True,
    )


for chunk in chat():
    print(chunk["message"]["content"], end="")
    # print(chunk)
    if chunk["done"]:
        print("End")
