from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

messages = []

while True:
    user_input = input("Hello sir,how can i help you? ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    messages.append({"role": "user", "content": user_input})
    
    # Keep only last 5 messages
    if len(messages) > 5:
        messages = messages[-5:]
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1:free",
        messages=messages
    )
    
    bot_response = response.choices[0].message.content
    print("Bot:", bot_response)
    
    messages.append({"role": "assistant", "content": bot_response})
