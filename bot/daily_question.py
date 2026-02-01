import random
import database
from openai import OpenAI
import config

db = database.Database()
client = OpenAI(api_key=config.openai_api_key)

themes = [
    "سوالی شاعرانه و خیالی درباره احساسات بپرس.",
    "سوالی بامزه و سبک درباره استرس روزمره بپرس.",
    "سوالی تأمل‌برانگیز درباره کنار آمدن با اضطراب بپرس.",
    "سوالی استعاری که استرس را به یک تصویر یا نماد تبدیل کند.",
    "سوالی مانند چک‌این روزانه با لحن صمیمی و آرام.",
]

def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def generate_unique_question():
    from openai_utils import ChatGPT  # Import here to avoid circular import
    chatgpt = ChatGPT(model="gpt-4o-mini")

    for _ in range(5):
        theme = random.choice(themes)
        system_prompt = f"""
        تو یک ربات تلگرام مهربون و خلاق هستی. وظیفه‌ات اینه که هر شب یک سوال کوتاه و صمیمی بپرسی که به کاربر کمک کنه درباره احساسات یا استرس خودش فکر کنه. 
        سوال را با لحن خودمانی، فارسی و به سبک موضوع زیر بنویس:

        موضوع: {theme}
        """

        try:
            answer, *_ = await chatgpt.send_message(system_prompt, dialog_messages=[], chat_mode="assistant")
            question = answer.strip()
            embedding = get_embedding(question)

            if not db.is_question_similar(embedding):
                db.save_question(question, embedding)
                return question

        except Exception as e:
            print(f"❌ Error: {e}")

    return "امشب نتونستم سوال تازه‌ای پیدا کنم. فردا دوباره امتحان می‌کنم."

async def send_daily_question(application):
    chat_id = config.daily_question_chat_id  # Add to config.py
    question = await generate_unique_question()
    await application.bot.send_message(chat_id=chat_id, text=question)
