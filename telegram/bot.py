import cv2
import numpy as np
import os
import time
import telebot
from food_recognizer import inference

API_TOKEN = os.environ.get('API_TOKEN')

bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(commands=['start'])
def welcome(message: telebot.types.Message) -> None:
    """
    Welcome function. Running when user press /start

    Parameters
    ----------
    message: telebot.types.Message
        This object represents a message.
    Returns
    -------
    None
    """
    bot.send_message(message.chat.id, 'Привет! я умею по картинке предсказывать, '
                                      'что за блюдо на ней изображено! Давай попробуем! '
                                      'Загружай свою!')


@bot.message_handler(content_types=['photo'])
def answer(message: telebot.types.Message) -> None:
    """
    Function get message from bot, and process it with neural network, to get prediction
    by image. Return prediction class for image

    Parameters
    ----------
    message: telebot.types.Message
        This object represents a message.

    Returns
    -------
    None
    """
    bot.send_message(message.chat.id, 'Анализирую изображение..Может занять несколько секунд')
    img = message.photo[-1]
    file_info = bot.get_file(img.file_id)
    photo = bot.download_file(file_info.file_path)
    img = cv2.imdecode(np.frombuffer(photo, np.uint8), 1)
    ans = inference(img)
    ans = f'Я думаю на изображение {ans}'
    bot.send_message(message.chat.id, ans)
    bot.send_message(message.chat.id, 'Проверить другое изображение еды?')


if __name__ == '__main__':
    while True:
        try:
            bot.polling(none_stop=True)
        except ():
            time.sleep(5)
