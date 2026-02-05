import io
import logging
import asyncio
from sched import scheduler
import traceback
import html
import json
from datetime import datetime
from openai import OpenAI

# Logging setup to save logs to a file
logging.basicConfig(
    level=logging.DEBUG,  # You can change this to ERROR or INFO depending on your needs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # This will output to the console
        logging.FileHandler("bot_logs.log")  # This will save the logs to a file
    ]
)

logger = logging.getLogger(__name__)  # Create a logger for the current module

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import daily_question

import telegram
from telegram.ext import ContextTypes
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils
import sys

print("Script started – configuring logging...", file=sys.stderr)
# setup
db = database.Database()

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help

🎨 Generate images from text prompts in <b>👩‍🎨 Artist</b> /mode
👥 Add bot to <b>group chat</b>: /help_group_chat
🎤 You can send <b>Voice Messages</b> instead of text

For more @yesbhautik
Powered by YesbhautikX 🚀
"""

HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

Instructions (see <b>video</b> below):
1. Add the bot to the group chat
2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
3. You're awesome!

To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
For example: "{bot_username} write a poem about Telegram"

Powered by YesbhautikX 🚀
"""

client = OpenAI(api_key=config.openai_api_key)

def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    logger.debug("Registering user: %s", user.id)
    
    if not db.check_if_user_exists(user.id):
        logger.info("User does not exist. Adding new user: %s", user.id)
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)
        logger.info("New user added and dialog started for user: %s", user.id)
    else:
        logger.info("User already exists: %s", user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        logger.info("No current dialog for user %s, starting a new dialog", user.id)
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)
        logger.debug("Semaphore created for user %s", user.id)

    if db.get_user_attribute(user.id, "current_model") is None:
        logger.info("No current model for user %s, setting default model", user.id)
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])
        logger.debug("Set default model for user %s", user.id)

    # Back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        logger.info("Old format for n_used_tokens found. Migrating data for user %s", user.id)
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)
        logger.debug("Migrated n_used_tokens for user %s", user.id)

    # Voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        logger.info("No transcription data found for user %s, initializing transcription seconds", user.id)
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # Image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        logger.info("No image generation data found for user %s, initializing generated images count", user.id)
        db.set_user_attribute(user.id, "n_generated_images", 0)



async def is_bot_mentioned(update: Update, context: CallbackContext):
    logger.debug("Checking if bot is mentioned in the message from user: %s", update.message.from_user.id)
    try:
        message = update.message

        # Check if the message is from a private chat
        if message.chat.type == "private":
            logger.info("Message is from a private chat, no need to check for bot mention.")
            return True

        # Check if the message mentions the bot by username
        if message.text is not None and ("@" + context.bot.username) in message.text:
            logger.info("Bot mentioned in the message: %s", message.text)
            return True

        # Check if the message is a reply to a message from the bot
        if message.reply_to_message is not None:
            if message.reply_to_message.from_user.id == context.bot.id:
                logger.info("Bot mentioned in the reply: %s", message.reply_to_message.text)
                return True
    except Exception as e:
        logger.error("Error while checking if bot is mentioned: %s", e)
        return True
    else:
        logger.debug("Bot was not mentioned.")
        return False


async def start_handle(update: Update, context: CallbackContext):
    logger.info("Start command received from user: %s", update.message.from_user.id)
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    logger.debug("Sending help message to user: %s", user_id)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)


async def help_handle(update: Update, context: CallbackContext):
    logger.info("Help command received from user: %s", update.message.from_user.id)
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    logger.debug("Sending help message to user: %s", user_id)
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def help_group_chat_handle(update: Update, context: CallbackContext):
    logger.info("Help group chat command received from user: %s", update.message.from_user.id)
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

    logger.debug("Sending group chat help message to user: %s", user_id)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    await update.message.reply_video(config.help_group_chat_video_path)


async def retry_handle(update: Update, context: CallbackContext):
    logger.info("Retry command received from user: %s", update.message.from_user.id)
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        logger.warning("No previous messages to retry for user: %s", user_id)
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # Last message was removed from the context

    logger.debug("Retrying with the last message for user: %s", user_id)
    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)

async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):

    if update.message.chat.type in ["group", "supergroup"]:
        user = update.message.from_user

        db.save_group_message(
            chat_id=update.message.chat.id,
            user_id=user.id,
            username=user.username or "",
            text=update.message.text or "",
            date=datetime.now()
        )


    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return



async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):

    if update.message.chat.type in ["group", "supergroup"]:
        user = update.message.from_user
        logger.info(f"Processing message from user {user.id} in group {update.message.chat.id}")

        db.save_group_message(
            chat_id=update.message.chat.id,
            user_id=user.id,
            username=user.username or "",
            text=update.message.text or "",
            date=datetime.now()
        )
        logger.info(f"Saved group message for user {user.id}.")

    # Check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        logger.info(f"Bot not mentioned in message from user {update.message.from_user.id}")
        return

    # Check if message is edited
    if update.edited_message is not None:
        logger.info(f"Message from user {update.message.from_user.id} was edited.")
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text
    logger.info(f"Received message from user {update.message.from_user.id}: {_message}")

    # Remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()
        logger.info(f"Removed bot mention from message: {_message}")

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context):
        return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    logger.info(f"User {user_id} is in {chat_mode} mode.")

    if chat_mode == "artist":
        logger.info(f"User {user_id} is in 'artist' mode, generating image.")
        await generate_image_handle(update, context, message=message)
        return

    async def message_handle_fn():
        # New dialog timeout
        if use_new_dialog_timeout:
            last_interaction_time = db.get_user_attribute(user_id, "last_interaction")
            if (datetime.now() - last_interaction_time).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
                logger.info(f"Started new dialog for user {user_id} due to timeout.")

        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")
        logger.info(f"Using model {current_model} for user {user_id}.")

        try:
            # Send placeholder message to user
            placeholder_message = await update.message.reply_text("...")
            logger.info(f"Sent placeholder message to user {user_id}.")

            # Send typing action
            await update.message.chat.send_action(action="typing")
            logger.info(f"Sending typing action for user {user_id}.")

            if _message is None or len(_message) == 0:
                logger.warning(f"User {user_id} sent an empty message.")
                await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]
            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            stream_prompt = '''
            شما یک ربات هوشمند و صمیمی هستید که به بچه‌ها کمک می‌کنید زمانی که احساس استرس یا نگرانی می‌کنند...
            '''

            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    _message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            prev_answer = ""
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                answer = answer[:4096]  # Telegram message limit

                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                    logger.info(f"Updated placeholder message for user {user_id} with new content.")
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)
                        logger.warning(f"Failed to update placeholder message for user {user_id}: {e}")

                await asyncio.sleep(0.01)  # Wait a bit to avoid flooding
                prev_answer = answer

            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )
            logger.info(f"Dialog updated for user {user_id} with new message.")

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            logger.info(f"Message handling was cancelled for user {user_id}.")
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(f"Error for user {user_id}: {error_text}")
            await update.message.reply_text(error_text)
            return

        # Send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
            logger.info(f"Message handling cancelled for user {user_id}.")
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]
                logger.info(f"User {user_id}'s task removed from user_tasks.")

async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        logger.info(f"User {user_id} is waiting for a reply to their previous message.")
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    # Log voice message transcription start
    logger.info(f"Transcribing voice message from user {user_id}.")
    transcribed_text = await openai_utils.transcribe_audio(buf)
    logger.info(f"Voice message transcribed for user {user_id}: {transcribed_text}")

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # Update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))
    logger.info(f"Updated transcribed seconds for user {user_id}: {voice.duration} seconds.")

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    logger.info(f"User {user_id} is requesting image generation with message: {message}")
    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images, size=config.image_size)
        logger.info(f"Generated {len(image_urls)} images for user {user_id}.")
    except openai.error.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            logger.warning(f"User {user_id}'s image request violated OpenAI's safety system.")
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            logger.error(f"Error while generating images for user {user_id}: {e}")
            raise

    # Token usage
    db.set_user_attribute(user_id, "n_generated_images", config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))
    logger.info(f"Updated generated image count for user {user_id}.")

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)
        logger.info(f"Sent generated image {i+1} to user {user_id}.")


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    logger.info(f"Starting new dialog for user {user_id}.")
    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)
    logger.info(f"Sent welcome message to user {user_id} for new dialog in {chat_mode} mode.")


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
        logger.info(f"Cancelled task for user {user_id}.")
    else:
        logger.info(f"No active task to cancel for user {user_id}.")
        await update.message.reply_text("<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")])
        elif is_last_page:
            keyboard.append([InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}")])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    logger.info(f"Displaying chat modes for user {user_id}.")
    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    if await is_previous_message_not_answered_yet(update.callback_query, context): return

    user_id = update.callback_query.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    query = update.callback_query
    await query.answer()

    page_index = int(query.data.split("|")[1])
    if page_index < 0:
        return

    logger.info(f"User {user_id} is navigating to page {page_index} of chat modes.")
    text, reply_markup = get_chat_mode_menu(page_index)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]
    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    logger.info(f"User {user_id} set chat mode to {chat_mode}.")

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    logger.info(f"Displaying settings for user {user_id}.")
    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    logger.info(f"User {user_id} set model to {model_key}.")

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = config.models["info"]["dalle-2"]["price_per_1_image"] * n_generated_images
    if n_generated_images != 0:
        details_text += f"- DALL·E 2 (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars

    logger.info(f"User {user_id} spent {total_n_spent_dollars:.03f}$, used {total_n_used_tokens} tokens.")

    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        logger.info(f"User {update.edited_message.from_user.id} tried to edit a message in private chat.")
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # Collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # Split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # Answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except Exception as e:
        logger.error(f"Error while handling error: {e}")
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")


async def opinionated_analysis_handle(update: Update, context: CallbackContext):
    if update.message.chat.type not in ["group", "supergroup"]:
        await update.message.reply_text("This command only works in group chats.")
        return

    chat_id = update.message.chat.id
    messages = db.get_last_group_messages(chat_id, limit=50)

    if not messages:
        logger.info(f"No messages found to analyze for group {chat_id}.")
        await update.message.reply_text("No messages found to analyze.")
        return

    # Merge messages
    combined_text = "\n".join(
        f"{msg.get('username', 'Unknown')}: {msg.get('text', '')}" 
        for msg in messages if msg.get("text")
    )

    combined_text = combined_text[:3000]  # Truncate if too long

    # 💬 Custom prompt for opinionated feedback
    prompt = f"""
    شما به عنوان یک ناظر با دقت، ۱۵۰ پیام آخر گروه چت را بررسی می‌کنید. لطفاً نظرتان را به‌طور واضح و بی‌پرده بیان کنید، بدون هیچ‌گونه سانسور:
    ...

    متن چت:
    {combined_text}
    """

    logger.info(f"Analyzing messages from group {chat_id}...")

    try:
        chatgpt = openai_utils.ChatGPT(model="gpt-4o")
        chat_mode = db.get_user_attribute(update.message.from_user.id, "current_chat_mode")
        response, *_ = await chatgpt.send_message(prompt, chat_mode=chat_mode)
        logger.info(f"Analysis complete for group {chat_id}, sending response.")
        await update.message.reply_text(f"🧠 خلاصه و نظر راجب 150 چت آخر:\n\n{response}")
    except Exception as e:
        logger.error(f"Failed to analyze messages in group {chat_id}: {e}")
        await update.message.reply_text(f"Failed to analyze messages: {e}")


async def id_handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    logger.info(f"Received update from chat {chat.id}: {update}")  # Log the full update
    await update.message.reply_text(f"Chat ID: {chat.id}")


async def post_init(application: Application):
    logger.info("🔧 Running post_init: setting commands and starting scheduler")

    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/mode", "Select chat mode"),
        BotCommand("/retry", "Re-generate response for previous query"),
        BotCommand("/balance", "Show balance"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/help", "Show help message"),
    ])
    logger.info("✔️ Bot commands set")

    # Start APScheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        ask_group,
        trigger="cron",
        hour="9,21",
        minute=0,
        args=[application]
    )
    scheduler.start()
    logger.info("📅 Scheduler started and job scheduled: ask_group at 9am and 9pm")



def main():
    logger.info("🔧 Starting bot setup...")

    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .post_init(post_init)
        .build()
    )

    logger.info("✅ Application built")

    # Add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    logger.info("✔️ Registered /start command")

    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))
    logger.info("✔️ Registered /help and /help_group_chat commands")

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    logger.info("✔️ Registered message handler")

    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))
    logger.info("✔️ Registered /retry, /new, /cancel commands")

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    logger.info("✔️ Registered voice message handler")

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))
    logger.info("✔️ Registered chat mode handlers")

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))
    logger.info("✔️ Registered settings handlers")

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))
    logger.info("✔️ Registered /balance handler")

    application.add_handler(CommandHandler("analyze", opinionated_analysis_handle, filters.ALL))
    application.add_handler(CommandHandler("ask", lambda u, c: asyncio.create_task(daily_question.ask_random(u, c))))
    application.add_handler(CommandHandler("id", id_handle))
    logger.info("✔️ Registered /analyze, /ask, and /id handlers")

    application.add_error_handler(error_handle)
    logger.info("✔️ Registered error handler")

    logger.info("🚀 Starting polling...")
    print("✅ Bot is running...")
    application.run_polling()


import random

async def ask_group(application):
    # chat_id = -1003501761776  #test group
    chat_id=1003675950022 #main group
    q = await daily_question.generate_unique_question()
    usernames = list(db.get_all_unique_usernames_in_group(chat_id))
    mentions = [f"@{u}" for u in random.sample(usernames, min(3, len(usernames)))]
    await application.bot.send_message(chat_id=chat_id, text=f"{q}\n\n📣 {' '.join(mentions)}")


if __name__ == "__main__":
    main()