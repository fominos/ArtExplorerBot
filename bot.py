import logging
import dill
import numpy as np
from PIL import Image
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
import faulthandler
from utils.image_processing import classify_image, analyze_style_and_emotion, generate_description

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)
from telegram.constants import ParseMode
IMG_SIZE = (224, 224)
CLASSIFICATION_MODEL_PATH = "models/mobilenet.pkl"
EMOTION_MODEL_PATH = "models/Artemis.pth"
BLIP_MODEL_PATH = "models/emotional_caption_model.dill"
BLIP_MODEL_ORIGINAL_PATH = "models/emotional_caption_model_original.dill"
giga = GigaChat(
    credentials="API_KEY",
    verify_ssl_certs=False,
    timeout=1200,
)
messages = [
    SystemMessage(
        content="Ты эксперт в сфере искусства. Помогаешь проанализировать картину. Используя только полученную информацию, сгенерируй развернутое эмоциональное описание картины на русском языке, избегая выдуманных деталей."
    )
]
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PHOTO = range(1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🎨 *Привет! Я — твой персональный искусствовед!* 🖼️\n\n"
        "Загрузи изображение произведения искусства, и я расскажу тебе:\n"
        "✅ Кто создал это произведение — *человек* или *ИИ*\n"
        "✅ Какой *художественный стиль* использован\n"
        "✅ Какие *эмоции* передает картина\n"
        "✅ И даже создам *эмоциональное описание*!\n\n"
        "✨ *Отправь мне изображение, и давай начнем!*"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    return PHOTO

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user = update.message.from_user
        photo_file = await update.message.photo[-1].get_file()
        image_path = "temp_image.jpg"
        await photo_file.download_to_drive(image_path)
        logger.info("Фотография %s: %s", user.id, f'{user.id}_photo.jpg')
        await update.message.reply_text(
            "✨ *Отлично!* Дай мне минуту, чтобы проанализировать изображение... 🕒",
            parse_mode=ParseMode.HTML
        )

        classification_result = classify_image(image_path, CLASSIFICATION_MODEL_PATH)
        style_pred, emotion_pred = analyze_style_and_emotion(image_path, EMOTION_MODEL_PATH)
        description = generate_description(image_path, BLIP_MODEL_PATH, BLIP_MODEL_ORIGINAL_PATH)
        content = (
            f"ВАЖНО: Не придумывай детали картины, позы людей, одежду, место и цветовую гамму, если они тебе неизвестны. \n"
            f"Опиши только то, что точно известно: стиль, эмоции и общее описание, избегая деталей о внешности и одежде.\n"
            f"Если что-то неясно, лучше оставь это без ответа.\n"
            f"Результаты анализа:\n"
            f"1. Художественный стиль: {style_pred}\n"
            f"2. Эмоция: {emotion_pred}\n"
            f"3. Описание: {description}")
        messages.append(HumanMessage(content=content))
        res = giga.invoke(messages)
        messages.append(res)
        print(description)
        response = (
            "🎨 <b>Результаты анализа:</b>\n\n"
            f"1. <b>Тип изображения:</b> {classification_result}\n"
            f"2. <b>Художественный стиль:</b> {style_pred}\n"
            f"3. <b>Эмоции:</b> {emotion_pred}\n"
            f"4. <b>Описание:</b> {res.content}\n\n"
            "✨ <b>Спасибо за использование бота!</b>\n\n"
            "🖼️ Хочешь загрузить еще одно изображение? Или напиши /cancel, чтобы завершить"
        )
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения. Попробуйте еще раз.")
    return PHOTO

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "✨ <b>До встречи!</b> 🖐️\n\n"
        "Чтобы продолжить, нажми команду /start",
        parse_mode=ParseMode.HTML
    )
    return ConversationHandler.END

def main() -> None:
    token = "BOT_TOKEN"
    if not token:
        raise ValueError("Токен бота не найден в переменных окружения!")

    application = (
        Application.builder()
        .token(token)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            PHOTO: [MessageHandler(filters.PHOTO, photo)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    faulthandler.enable()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main()
