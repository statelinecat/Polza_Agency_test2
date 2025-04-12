import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional, Dict
from dotenv import load_dotenv
from googletrans import Translator
import requests
import logging
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация приложения
DATABASE_FILE = "complaints.db"
SENTIMENT_API_URL = "https://api.apilayer.com/sentiment/analysis"
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY")
SPAM_API_URL = "https://api.api-ninjas.com/v1/spamcheck"
SPAM_API_KEY = os.getenv("SPAM_API_KEY")
GEOLOCATION_API_URL = "http://ip-api.com/json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_TIMEOUT = 20

# Инициализация OpenAI клиента
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

logger.info("Application configuration loaded")


# Модели данных
class Complaint(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000, example="Не приходит SMS-код")


class ComplaintResponse(BaseModel):
    id: int
    status: str
    sentiment: str
    category: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


# Инициализация базы данных
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sentiment TEXT,
            category TEXT DEFAULT 'другое',
            geolocation TEXT
        )
        """)
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise RuntimeError(f"Database initialization failed: {str(e)}")
    finally:
        if conn:
            conn.close()


async def translate_to_english(text: str) -> str:
    """Перевод текста на английский"""
    translator = Translator()
    try:
        translated = await translator.translate(text, src='ru', dest='en')
        logger.info(f"Translated text: {translated.text}")
        return translated.text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text


async def analyze_sentiment(text: str) -> str:
    """Анализ тональности через внешний API"""
    try:
        translated_text = await translate_to_english(text)
        response = requests.post(
            SENTIMENT_API_URL,
            headers={"apikey": SENTIMENT_API_KEY},
            json={"text": translated_text},
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            sentiment = result.get("sentiment", "unknown").lower()
            return sentiment if sentiment in ["positive", "negative", "neutral"] else "unknown"

        logger.warning(f"Sentiment API error: {response.status_code}")
        return "unknown"
    except requests.exceptions.RequestException as e:
        logger.error(f"Sentiment API request failed: {e}")
        return "unknown"


def check_spam(text: str) -> bool:
    """Проверка текста на спам через внешний API"""
    if not SPAM_API_KEY:
        logger.warning("Spam API key not found! Skipping spam check")
        return False

    try:
        response = requests.get(
            SPAM_API_URL,
            headers={"X-Api-Key": SPAM_API_KEY},
            params={"text": text},
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("is_spam", False)

        logger.warning(f"Spam API error: {response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Spam API request failed: {e}")
        return False


async def detect_category_with_ai(text: str) -> str:
    """Определение категории жалобы с использованием OpenAI API"""
    if not OPENAI_API_KEY or not openai_client:
        logger.warning("OpenAI API not available! Using default category 'другое'")
        return "другое"

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ты классификатор жалоб. Отвечай строго одним словом: 'техническая', 'оплата' или 'другое'."
                },
                {
                    "role": "user",
                    "content": f"Определи категорию жалобы: '{text}'. Ответь только одним словом: техническая, оплата или другое."
                }
            ],
            temperature=0.0,
            max_tokens=10
        )

        category = response.choices[0].message.content.strip().lower()
        logger.info(f"OpenAI category detection result: {category}")

        if category in ["техническая", "оплата"]:
            return category
        return "другое"

    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}. Using default category 'другое'")
        return "другое"


def get_geolocation(ip: str) -> Dict[str, str]:
    """Получение геолокации по IP"""
    try:
        response = requests.get(f"{GEOLOCATION_API_URL}/{ip}", timeout=API_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return {
                    "country": data.get("country"),
                    "city": data.get("city"),
                    "timezone": data.get("timezone")
                }
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Geolocation API request failed: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up... Initializing database")
    init_db()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="Complaint Processing API",
    description="API для обработки жалоб клиентов с анализом тональности и категоризацией с помощью ИИ",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/complaints/",
    response_model=ComplaintResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректные данные"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def create_complaint(complaint: Complaint, request: Request):
    try:
        if check_spam(complaint.text):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Текст содержит спам"
            )

        sentiment = await analyze_sentiment(complaint.text)
        category = await detect_category_with_ai(complaint.text)
        client_ip = request.client.host
        geolocation = get_geolocation(client_ip)
        geolocation_str = ", ".join(f"{k}: {v}" for k, v in geolocation.items()) if geolocation else None

        conn = None
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute(
                """INSERT INTO complaints (text, sentiment, category, geolocation)
                   VALUES (?, ?, ?, ?)""",
                (complaint.text, sentiment, category, geolocation_str)
            )
            complaint_id = cursor.lastrowid
            conn.commit()

            cursor.execute(
                """SELECT id, status, sentiment, category 
                   FROM complaints WHERE id = ?""",
                (complaint_id,)
            )
            record = cursor.fetchone()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Не удалось получить созданную жалобу"
                )

            return {
                "id": record[0],
                "status": record[1],
                "sentiment": record[2],
                "category": record[3] if record[3] != "другое" else None
            }
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка базы данных: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.get(
    "/complaints/{complaint_id}",
    response_model=ComplaintResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Жалоба не найдена"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def get_complaint(complaint_id: int):
    try:
        conn = None
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute(
                """SELECT id, status, sentiment, category 
                   FROM complaints WHERE id = ?""",
                (complaint_id,)
            )
            record = cursor.fetchone()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Жалоба не найдена"
                )

            return {
                "id": record[0],
                "status": record[1],
                "sentiment": record[2],
                "category": record[3] if record[3] != "другое" else None
            }
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка базы данных: {str(e)}"
            )
        finally:
            if conn:
                conn.close()
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)