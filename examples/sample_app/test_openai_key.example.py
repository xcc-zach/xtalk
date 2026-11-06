from openai import OpenAI

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # 在这里填写你的 API key

response = client.responses.create(
    model="gpt-4.1-mini",
    input="请只回复 OK。",
)
print(response.output_text)
