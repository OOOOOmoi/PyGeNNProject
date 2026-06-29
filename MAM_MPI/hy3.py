from openai import OpenAI

client = OpenAI(
    api_key="sk-hy3-admin-001",
    base_url="http://172.31.1.10:8080/v1",
)

response = client.chat.completions.create(
    model="hy3-preview",
    messages=[
        {"role": "user", "content": "你好"}
    ],
)

print(response.choices[0].message.content)