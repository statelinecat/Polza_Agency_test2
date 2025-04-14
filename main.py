import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import requests
from googletrans import Translator
from openai import AsyncOpenAI
import pytz
import re

# ===== Конфигурация приложения =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
load_dotenv()


class Config:
    DATABASE_FILE = os.getenv("DATABASE_FILE", "complaints.db")
    SENTIMENT_API_URL = "https://api.apilayer.com/sentiment/analysis"
    SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY")
    SPAM_API_URL = "https://api.api-ninjas.com/v1/spamcheck"
    SPAM_API_KEY = os.getenv("SPAM_API_KEY")
    GEOLOCATION_API_URL = "http://ip-api.com/json"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    API_TIMEOUT = 20
    TIMEZONE = pytz.timezone(os.getenv("TIMEZONE", "Europe/Moscow"))
    CATEGORIZATION_METHOD = os.getenv("CATEGORIZATION_METHOD", "local").lower()  # 'openai' или 'local'


translator = Translator()
openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else None


# ===== Модели данных =====
class ComplaintBase(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000, example="Не приходит SMS-код")


class ComplaintCreate(ComplaintBase):
    pass


class ComplaintResponse(BaseModel):
    id: int
    text: str
    status: str
    sentiment: Optional[str] = None
    category: Optional[str] = None
    geolocation: Optional[str] = None
    timestamp: Optional[datetime] = None


class StatusUpdate(BaseModel):
    id: int = Field(..., example=1)
    status: str = Field(..., example="closed")


class ErrorResponse(BaseModel):
    detail: str


# ===== Вспомогательные функции =====
def init_db():
    try:
        with sqlite3.connect(Config.DATABASE_FILE) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sentiment TEXT,
                category TEXT DEFAULT 'другое',
                geolocation TEXT
            )""")
            logger.info("Database initialized")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise RuntimeError(f"Database init failed: {str(e)}")


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None

    try:
        # Пробуем ISO формат
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.astimezone(Config.TIMEZONE)
    except ValueError:
        try:
            # Пробуем UNIX timestamp
            dt = datetime.fromtimestamp(int(dt_str))
            return dt.astimezone(Config.TIMEZONE)
        except ValueError:
            return None


async def translate_to_english(text: str) -> str:
    try:
        translated = await translator.translate(text, src='ru', dest='en')
        return translated.text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text


async def analyze_sentiment(text: str) -> str:
    try:
        translated_text = await translate_to_english(text)
        response = requests.post(
            Config.SENTIMENT_API_URL,
            headers={"apikey": Config.SENTIMENT_API_KEY},
            json={"text": translated_text},
            timeout=Config.API_TIMEOUT
        )
        if response.status_code == 200:
            sentiment = response.json().get("sentiment", "unknown").lower()
            return sentiment if sentiment in ["positive", "negative", "neutral"] else "unknown"
        logger.warning(f"Sentiment API error: {response.status_code}")
        return "unknown"
    except requests.exceptions.RequestException as e:
        logger.error(f"Sentiment API error: {e}")
        return "unknown"


def check_spam(text: str) -> bool:
    if not Config.SPAM_API_KEY:
        logger.warning("Spam API key missing")
        return False

    try:
        response = requests.get(
            Config.SPAM_API_URL,
            headers={"X-Api-Key": Config.SPAM_API_KEY},
            params={"text": text},
            timeout=Config.API_TIMEOUT
        )
        return response.json().get("is_spam", False) if response.status_code == 200 else False
    except requests.exceptions.RequestException as e:
        logger.error(f"Spam API error: {e}")
        return False


def detect_category_local(text: str) -> str:
    """
    Локальная функция для определения категории жалобы на основе ключевых слов.
    Возвращает 'техническая', 'оплата' или 'другое'.
    """
    # Нормализуем текст: нижний регистр, удаляем лишние пробелы и знаки препинания
    normalized_text = re.sub(r'[^\w\s]', '', text.lower().strip())

    # Ключевые слова для технических проблем (с приоритетом)
    tech_keywords = [
        'не работает', 'ошибка', 'сбой', 'технический', 'техническая',
        'баг', 'глюк', 'sms', 'смс', 'код', 'верификац', 'авторизац',
        'загрузк', 'приложен', 'приложение', 'сайт', 'страниц', 'интернет',
        'соединен', 'подключен', 'сервер', 'сообщен', 'уведомлен', 'не приходит',
        'не отправляется', 'не получаю'
    ]

    # Ключевые слова для проблем с оплатой (с приоритетом)
    payment_keywords = [
        'оплат', 'платеж', 'деньг', 'средств', 'перевод', 'карт',
        'счет', 'банк', 'списан', 'возврат', 'денежн', 'валюта',
        'комисс', 'чек', 'транзакц', 'refund', 'купил', 'покупк',
        'не проходит', 'не уходит', 'не зачисляется'
    ]

    # Проверяем наличие ключевых слов с учетом частичного совпадения
    is_tech = any(re.search(rf'\b{re.escape(kw)}', normalized_text) for kw in tech_keywords)
    is_payment = any(re.search(rf'\b{re.escape(kw)}', normalized_text) for kw in payment_keywords)

    # Определяем категорию с учетом приоритетов
    if is_payment and not is_tech:
        return "оплата"
    elif is_tech and not is_payment:
        return "техническая"
    elif is_tech and is_payment:
        # Если есть оба типа ключевых слов, проверяем более конкретные совпадения
        payment_score = sum(1 for kw in payment_keywords if re.search(rf'\b{re.escape(kw)}', normalized_text))
        tech_score = sum(1 for kw in tech_keywords if re.search(rf'\b{re.escape(kw)}', normalized_text))
        return "оплата" if payment_score > tech_score else "техническая"
    else:
        return "другое"


async def detect_category_openai(text: str) -> str:
    """Определение категории с помощью OpenAI"""
    if not Config.OPENAI_API_KEY or not openai_client:
        logger.warning("OpenAI API unavailable")
        return "другое"

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Классифицируй жалобы: 'техническая', 'оплата' или 'другое'"},
                {"role": "user", "content": f"Категория жалобы: '{text}'"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip().lower()
        return category if category in ["техническая", "оплата"] else "другое"
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "другое"


async def detect_category(text: str) -> str:
    """Выбор метода категоризации в зависимости от конфигурации"""
    if Config.CATEGORIZATION_METHOD == "openai":
        return await detect_category_openai(text)
    else:
        return detect_category_local(text)


def get_geolocation(ip: str) -> Dict[str, str]:
    try:
        response = requests.get(f"{Config.GEOLOCATION_API_URL}/{ip}", timeout=Config.API_TIMEOUT)
        if response.status_code == 200 and response.json().get("status") == "success":
            return {
                "country": response.json().get("country"),
                "city": response.json().get("city"),
                "timezone": response.json().get("timezone")
            }
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Geolocation error: {e}")
        return {}


# ===== Инициализация FastAPI =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app...")
    init_db()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="Complaint API",
    description="API для обработки жалоб",
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


# ===== Эндпоинты =====
@app.post(
    "/complaints/",
    response_model=ComplaintResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректные данные"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def create_complaint(complaint: ComplaintCreate, request: Request):
    if check_spam(complaint.text):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Текст содержит спам"
        )

    try:
        sentiment = await analyze_sentiment(complaint.text)
        category = await detect_category(complaint.text)
        geolocation = get_geolocation(request.client.host)
        geolocation_str = ", ".join(f"{k}: {v}" for k, v in geolocation.items()) if geolocation else None

        with sqlite3.connect(Config.DATABASE_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO complaints (text, sentiment, category, geolocation)
                   VALUES (?, ?, ?, ?)""",
                (complaint.text, sentiment, category, geolocation_str)
            )
            complaint_id = cursor.lastrowid

            cursor.execute(
                """SELECT id, text, status, sentiment, category, geolocation, timestamp
                   FROM complaints WHERE id = ?""",
                (complaint_id,)
            )
            record = cursor.fetchone()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Не удалось получить созданную жалобу"
                )

            return ComplaintResponse(
                id=record[0],
                text=record[1],
                status=record[2],
                sentiment=record[3],
                category=record[4],
                geolocation=record[5],
                timestamp=record[6]
            )

    except sqlite3.Error as e:
        logger.error(f"DB error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка БД: {str(e)}"
        )


@app.get(
    "/complaints/",
    response_model=List[ComplaintResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Жалобы не найдены"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def list_complaints(
        status_filter: Optional[str] = Query(None, description="Фильтр по статусу"),
        since: Optional[str] = Query(None, description="Дата в ISO формате или UNIX timestamp")
):
    try:
        since_dt = parse_datetime(since)

        with sqlite3.connect(Config.DATABASE_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT id, text, status, sentiment, category, geolocation, timestamp
                FROM complaints
                WHERE 1=1
            """
            params = []

            if status_filter:
                query += " AND status = ?"
                params.append(status_filter)

            if since_dt:
                query += " AND timestamp >= ?"
                params.append(since_dt.strftime("%Y-%m-%d %H:%M:%S"))

            logger.info(f"Executing query: {query} with params: {params}")
            cursor.execute(query, params)
            records = cursor.fetchall()

            if not records:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Жалобы не найдены"
                )

            return [
                ComplaintResponse(
                    id=row["id"],
                    text=row["text"],
                    status=row["status"],
                    sentiment=row["sentiment"],
                    category=row["category"],
                    geolocation=row["geolocation"],
                    timestamp=row["timestamp"]
                ) for row in records
            ]

    except sqlite3.Error as e:
        logger.error(f"DB error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка БД: {str(e)}"
        )


@app.patch(
    "/complaints/status",
    response_model=ComplaintResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Неверные данные"},
        404: {"model": ErrorResponse, "description": "Жалоба не найдена"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def update_complaint_status(update: StatusUpdate):
    try:
        with sqlite3.connect(Config.DATABASE_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Проверка существования жалобы
            cursor.execute("SELECT id FROM complaints WHERE id = ?", (update.id,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Жалоба не найдена"
                )

            # Обновление статуса
            cursor.execute(
                "UPDATE complaints SET status = ? WHERE id = ?",
                (update.status, update.id)
            )
            conn.commit()

            # Получение обновленных данных
            cursor.execute(
                """SELECT id, text, status, sentiment, category, geolocation, timestamp
                   FROM complaints WHERE id = ?""",
                (update.id,)
            )
            record = cursor.fetchone()

            return ComplaintResponse(
                id=record["id"],
                text=record["text"],
                status=record["status"],
                sentiment=record["sentiment"],
                category=record["category"],
                geolocation=record["geolocation"],
                timestamp=record["timestamp"]
            )

    except sqlite3.Error as e:
        logger.error(f"DB error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка БД: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)