import sqlite3
from datetime import datetime
import random

# Подключение к базе данных (та же, что используется в основном приложении)
DB_FILE = "complaints.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Тестовые данные
complaints = [
    {"text": "Не приходит SMS для подтверждения входа", "category": "техническая"},
    {"text": "Оплата не проходит уже 2 дня", "category": "оплата"},
    {"text": "Ошибочно списали деньги со счета", "category": "оплата"},
    {"text": "Приложение постоянно вылетает при открытии", "category": "техническая"},
    {"text": "Не могу авторизоваться в личном кабинете", "category": "техническая"},
    {"text": "Не пришел чек за последнюю транзакцию", "category": "оплата"},
    {"text": "Сайт грузится очень медленно", "category": "техническая"},
    {"text": "Не могу привязать новую карту", "category": "оплата"},
    {"text": "Потерял доступ к аккаунту", "category": "техническая"},
    {"text": "Двойное списание средств за покупку", "category": "оплата"},
    {"text": "Хочу поблагодарить за отличный сервис", "category": "другое"},
    {"text": "Где можно посмотреть историю платежей?", "category": "другое"}
]

# Заполнение базы данных
for idx, complaint in enumerate(complaints, 1):
    cursor.execute(
        """INSERT INTO complaints 
        (text, status, category, timestamp) 
        VALUES (?, ?, ?, ?)""",
        (
            complaint["text"],
            random.choice(["open", "closed"]),  # Только open или closed
            complaint["category"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Текущее время
        )
    )
    print(f"Добавлена жалоба {idx}/{len(complaints)}: {complaint['text']}")

conn.commit()
conn.close()
print("База данных успешно заполнена тестовыми данными!")