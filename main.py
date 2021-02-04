import telebot
import codecs
from process_request import request
from telebot import types
from model import bag_of_word, bag_of_word_euclidian
from process_request import request_elmo
from telegram.ext.dispatcher import run_async
from process_request import request_euclidian
dict_prev_mes = {}
with codecs.open('start.txt', encoding='utf-8') as fin:  # вывод стартового сообщения при получении команды
    s = fin.read()
with codecs.open('help.txt', encoding='utf-8') as fin:  # вывод сообщения помощи при получении команды
    h = fin.read()
key_cons = types.InlineKeyboardButton(text='Консультация word2vec', callback_data='model')  # кнопка консультации - вызов
key_cons2 = types.InlineKeyboardButton(text='Консультация bagofword', callback_data='model2')  # кнопка консультации - вызов
@bot.callback_query_handler(func=lambda call: True)
def ans(call):   # функция для работы кнопки "Консультация"
    global dict_prev_mes
    if call.data == 'model':
        dict_prev_mes[call.message.chat.id] = '/cons_word2vec'
        bot.send_message(call.message.chat.id, 'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    else:
        dict_prev_mes[call.message.chat.id] = '/cons_bagofword'
        bot.send_message(call.message.chat.id,'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global dict_prev_mes
    print(message.text)
    if message.text == '/start':                #вывод стартового сообщения и кнопки
        dict_prev_mes[message.from_user.id] = message.text
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(key_cons)
        keyboard.add(key_cons2)
        bot.send_message(message.from_user.id, s, reply_markup=keyboard)
    elif message.text == '/help':               #Вывод сообщения при введение команды /help
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id, h)
    elif message.text == '/cons_word2vec':               #Вывод сообщения при введение команды /cons_word2vec
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id,
                         'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    elif message.text == '/cons_word2vec_euclidian':               #Вывод сообщения при введение команды /cons_word2vec_euclidian
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id,
                         'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    elif message.text == '/cons_bagofword':  # Вывод сообщения при введение команды /cons_bagofword
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id,
                         'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    elif message.text == '/cons_bagofword_euclidian':  # Вывод сообщения при введение команды /cons_bagofword_euclidian
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id,
                         'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    elif message.text == '/cons':               #Вывод сообщения при введение команды /cons
        dict_prev_mes[message.from_user.id] = message.text
        bot.send_message(message.from_user.id,
                         'Опишите чем Вы увлекаетесь, какие у Вас есть навыки и личностные качества')
    elif dict_prev_mes[message.from_user.id] == '/cons':                 # Запуск анализа естественного запроса
        list_of_profession = request(message.text)
        if list_of_profession == 'Не могу ничего подсказать, опишите подробнее':
            bot.send_message(message.from_user.id,str(list_of_profession))
        else:
            bot.send_message(message.from_user.id, 'Основываясь на полученных данных Вам больше всего подойдут:')
            for i in list_of_profession:
                bot.send_message(message.from_user.id, i + '\n')
            dict_prev_mes[message.from_user.id] = message.text
    elif dict_prev_mes[message.from_user.id] == '/cons_word2vec_euclidian':                 # Запуск анализа естественного запроса
        list_of_profession = request_euclidian(message.text)
        if list_of_profession == 'Не могу ничего подсказать, опишите подробнее':
            bot.send_message(message.from_user.id,str(list_of_profession))
        else:
            bot.send_message(message.from_user.id, 'Основываясь на полученных данных Вам больше всего подойдут:')
            for i in list_of_profession:
                bot.send_message(message.from_user.id, i + '\n')
            dict_prev_mes[message.from_user.id] = message.text
    elif dict_prev_mes[message.from_user.id] == '/cons_word2vec':                 # Запуск анализа естественного запроса
        list_of_profession = request(message.text)
        if list_of_profession == 'Не могу ничего подсказать, опишите подробнее':
            bot.send_message(message.from_user.id,str(list_of_profession))
        else:
            bot.send_message(message.from_user.id, 'Основываясь на полученных данных Вам больше всего подойдут:')
            for i in list_of_profession:
                bot.send_message(message.from_user.id, i + '\n')
            dict_prev_mes[message.from_user.id] = message.text
    elif dict_prev_mes[message.from_user.id] == '/cons_bagofword':                 # Запуск анализа естественного запроса
        list_of_profession = bag_of_word(message.text)
        if list_of_profession == 'Не могу ничего подсказать, опишите подробнее':
            bot.send_message(message.from_user.id,str(list_of_profession))
        else:
            bot.send_message(message.from_user.id, 'Основываясь на полученных данных Вам больше всего подойдут:')
            for i in list_of_profession:
                bot.send_message(message.from_user.id, i + '\n')
            dict_prev_mes[message.from_user.id] = message.text
    elif dict_prev_mes[message.from_user.id] == '/cons_bagofword_euclidian':                 # Запуск анализа естественного запроса
        list_of_profession = bag_of_word_euclidian(message.text)
        if list_of_profession == 'Не могу ничего подсказать, опишите подробнее':
            bot.send_message(message.from_user.id,str(list_of_profession))
        else:
            bot.send_message(message.from_user.id, 'Основываясь на полученных данных Вам больше всего подойдут:')
            for i in list_of_profession:
                bot.send_message(message.from_user.id, i + '\n')
            dict_prev_mes[message.from_user.id] = message.text
    else:
        bot.send_message(message.from_user.id, 'Если Вы хотели рассказать о себе для получения подходящих профессий используйте сначала команды из меню help')
        dict_prev_mes[message.from_user.id] = message.text
bot.polling(none_stop=True, interval=0)
