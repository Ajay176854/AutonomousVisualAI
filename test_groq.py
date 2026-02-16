from groq import Groq

client = Groq(api_key="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")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Explain AI in one sentence"}
    ],
)

print(response.choices[0].message.content)


