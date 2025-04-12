import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Загрузка API-ключа
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

print(os.getenv("OPENAI_API_KEY"))

# Пример запроса
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Ты помощник для тестирования API."},
        {"role": "user", "content": "Привет! Как дела?"}
    ],
    max_tokens=50,
    temperature=0.7
)

# Вывод ответа
print(response.choices[0].message.content)