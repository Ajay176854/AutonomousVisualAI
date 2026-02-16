from groq import Groq

# initialize client
client = Groq(api_key="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")

def ask_llm(objects, text, question):

    prompt = f"""
You are a visual assistant.

Objects detected in the image:
{objects}

Extracted text from the image:
{text}

User question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content
