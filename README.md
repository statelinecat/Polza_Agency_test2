# Polza_Agency_test2
 Тестовое задание от Polza Agency

### FastAPI-сервис для обработки клиентских жалоб с анализом тональности, категоризацией, проверкой на спам и определением геолокации по IP с использованием публичных API, определение категории жалобы с помощью ИИ
.

### 📌 Функциональность
Принимает POST-запросы с текстом жалобы.

### Выполняет:
Анализ тональности (Sentiment Analysis via APILayer).
Категоризацию жалоб (технические / платежные / прочие).
Проверку на спам (Spam Check via API Ninjas, опционально).
Определение геолокации по IP (через ip-api.com).
Сохраняет жалобу в SQLite.
Автоматически определяет категорию жалобы (техническая, оплата, другое) с использованием: OpenAI API (GPT-3.5 Turbo)
Возвращает JSON-ответ с данными жалобы.
Обрабатывает ошибки внешних API.

### Установка и запуск

#### 1. Клонировать репозиторий
git clone https://github.com/statelinecat/Polza_Agency_test2.git
cd Polza_Agency_test2

#### 2. Создать и активировать виртуальное окружение (опционально)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

#### 3. Установить зависимости
pip install -r requirements.txt

#### 4. Создать .env файл с API ключами
SENTIMENT_API_KEY=ваш_ключ_apilayer
SPAM_API_KEY=ваш_ключ_api_ninjas
OPENAI_API_KEY=ваш_ключ_openai  # опционально
TIMEZONE=Europe/Moscow  # ваша временная зона

#### 5. Запустить сервер
uvicorn main:app --reload

### 🔗 Документация API
Доступна по адресу:
http://176.108.248.61:8002/docs

### 📮 Примеры запросов
#### Создание жалобы
curl -X POST http://localhost:8000/complaints/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Не приходит SMS-код"}'

#### Ответ:

{
  "id": 1,
  "status": "open",
  "sentiment": "negative",
  "category": "техническая"
}

### 🧪 Тестирование
Используйте Postman или curl для тестов. Примеры выше.

### ❗ Обработка ошибок
При сбое API анализа тональности: sentiment = "unknown".
Если определение категории невозможно, сохраняет category = "другое".
При подозрении на спам: возвращается 400 Bad Request.
При ошибках базы данных и сервера: 500 Internal Server Error.

### 📝 Лицензия
MIT License. Используйте и модифицируйте свободно.