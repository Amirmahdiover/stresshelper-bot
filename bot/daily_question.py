import random
import database
from openai import OpenAI
import config
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode

db = database.Database()
client = OpenAI(api_key=config.openai_api_key)


async def generate_unique_question():
    from openai_utils import ChatGPT  # Import here to avoid circular import
    chatgpt = ChatGPT(model="gpt-4o-mini")

    # Retrieve previous questions from the database
    previous_questions = db.get_all_questions()

    # Prepare the context from previous questions to ensure uniqueness
    previous_questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(previous_questions)])

    # System prompt asking for a new, unique question
    system_prompt = f"""
    تو یک ربات تلگرام مهربون و خلاق هستی. وظیفه‌ات اینه که هر شب یک سوال کوتاه و صمیمی بپرسی که به کاربر کمک کنه درباره احساسات یا استرس خودش فکر کنه.
    سوال را با لحن خودمانی و فارسی بنویس. همچنین از سوالات قبلی که پرسیده‌ای به هیچ وجه شبیه سوال جدید نباشد:

    سوالات قبلی:
    {previous_questions_text}

    لطفاً یک سوال جدید بنویس که هیچ شباهتی به سوالات قبلی نداشته باشد.
    """

    try:
        answer, *_ = await chatgpt.send_message(system_prompt, dialog_messages=[], chat_mode="assistant")
        question = answer.strip()

        # Check if the new question is unique by making sure it's not in the previous questions
        if question not in previous_questions_text:
            # Save the new question to the database and return it
            db.save_question(question)
            return question
        else:
            # If the question is too similar, return a fallback message
            return "امشب نتونستم سوال تازه‌ای پیدا کنم. فردا دوباره امتحان می‌کنم."
    except Exception as e:
        print(f"❌ Error: {e}")
        return "امشب نتونستم سوال تازه‌ای پیدا کنم. فردا دوباره امتحان می‌کنم."
    


async def ask_random(update: Update, context: CallbackContext):
    chat = update.effective_chat
    if chat.type not in ["group", "supergroup"]:
        await update.message.reply_text("این دستور فقط در گروه‌ها کار می‌کند.")
        return

    question = await generate_unique_question()

    # Use all-time usernames
    usernames = db.get_all_unique_usernames_in_group(chat.id)

    if usernames:
        selected = random.sample(usernames, min(3, len(usernames)))
        mentions = [f"@{u}" for u in selected]
        text = f"{question}\n\nبیا صحبت کنیم! {' '.join(mentions)}"
    else:
        text = question

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
