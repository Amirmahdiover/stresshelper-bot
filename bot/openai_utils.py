from openai import OpenAI
import config

client = OpenAI(api_key=config.openai_api_key)

def build_messages(message, dialog_messages, chat_mode):
    messages = [{"role": "system", "content": config.chat_modes[chat_mode]["prompt_start"]}]
    for m in dialog_messages:
        messages.append({"role": "user", "content": m["user"]})
        messages.append({"role": "assistant", "content": m["bot"]})
    messages.append({"role": "user", "content": message})
    return messages

class ChatGPT:
    def __init__(self, model="gpt-4o"):
        self.model = model

        # Add system_prompt as an optional parameter
    async def send_message(self, message, dialog_messages=[], chat_mode="assistant", system_prompt=None):
        messages = build_messages(message, dialog_messages, chat_mode)

        # Inject system prompt if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = response.choices[0].message.content
        usage = response.usage
        return answer, (usage.prompt_tokens, usage.completion_tokens), 0


    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant", system_prompt=None):
        messages = build_messages(message, dialog_messages, chat_mode)

        # Inject system prompt if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        answer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            answer += delta
            yield "not_finished", answer, (0, 0), 0
        yield "finished", answer, (0, 0), 0

async def transcribe_audio(audio_file) -> str:
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return response.text

async def generate_images(prompt, n_images=4, size="512x512"):
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=n_images,
        size=size
    )
    return [img.url for img in response.data]

async def is_content_acceptable(prompt):
    response = client.moderations.create(input=prompt)
    return not any(response.results[0].categories.values())
