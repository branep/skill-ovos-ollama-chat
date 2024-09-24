from ollama import Client

client = Client(host="http://ollama.lan")
response = client.chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ],
    keep_alive=-1,
    stream=True,
)

for chunk in response:
    print (chunk['message']['content'], end='', flush=True)
