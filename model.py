import re
from math import sqrt

import gensim
import pandas as pd
import json
import pymorphy2
from scipy.spatial import distance
from vector_create import vector_create_by_str
import tokenization
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from summa.summarizer import summarize
import tensorflow as tf
import tensorflow_hub as hub
from vector_create import elmo_vector_create

folder = 'multi_cased_L-12_H-768_A-12'
config_path = folder + '/bert_config.json'
checkpoint_path = folder + '/bert_model.ckpt'
vocab_path = folder + '/vocab.txt'


def word2vec():
    df = pd.read_csv('dataframe.csv')  # загружаем частично подготовленный data frame
    vectors_def = []  # создаем список векторов определений
    for i in range(1, 130, 1):  # идем по строкам data frame
        final_vector = vector_create_by_str(df['2'].values[i])  # для каждого определения строим вектор
        for i in range(len(final_vector)):
            final_vector[i] = str(final_vector[i])  # приводим float->string для нормальной работы json
        vectors_def.append(final_vector)  # добавляем вектор определения в список векторов определений
    f = open('model_vectors.txt', 'w')  # открываем файл для записи векторов определений
    json.dump(vectors_def, f)  # записываем json наш список списков векторов
    f.close()  # закрываем файл

def bag_of_word(input_request):  # DONE
    morph = pymorphy2.MorphAnalyzer()  # подключаем анализатор слов
    df = pd.read_csv('dataframe.csv')  # загружаем частично подготовленный data frame
    all_words = ""
    for i in range(1, 130, 1):
        all_words += df['1'].values[i] + ' ' + df['2'].values[i]
    buf = gensim.utils.simple_preprocess(all_words)  # очищаем запрос от второстепенных данных
    buf2 = ' '.join(buf)  # приводим полученный вид запроса в строку
    buf3 = re.split(r'[ \–\, .]+', buf2)  # удаляем знаки пунктуации
    buf4 = []
    for j in buf3:
        try:
            buf4.append(morph.parse(j)[0].normal_form)
        except:  # если этого не получается сделать пропускаем данное слово
            pass
    dict_of_words = {}
    i = 0
    for j in buf4:
        if not j in dict_of_words:
            dict_of_words.update({j: i})
            i += 1
    vectors_of_definitions = []
    for i in range(0, 130, 1):
        buf = []
        for j in range(len(dict_of_words)):
            buf.append(0)
        definition = df['2'].values[i]
        buf2 = gensim.utils.simple_preprocess(definition)  # очищаем запрос от второстепенных данных
        buf3 = ' '.join(buf2)  # приводим полученный вид запроса в строку
        buf4 = re.split(r'[ \–\, .]+', buf3)
        buf5 = []
        for j in buf4:
            try:
                buf5.append(morph.parse(j)[0].normal_form)
            except:  # если этого не получается сделать пропускаем данное слово
                pass
        for j in buf5:
            if j in dict_of_words:
                buf[dict_of_words.get(j)] += 1
        vectors_of_definitions.append([df['1'].values[i], buf])
    buf = gensim.utils.simple_preprocess(input_request)  # очищаем запрос от второстепенных данных
    buf2 = ' '.join(buf)  # приводим полученный вид запроса в строку
    buf3 = re.split(r'[ \–\, .]+', buf2)  # удаляем знаки пунктуации
    buf4 = []
    for j in buf3:
        try:
            buf4.append(morph.parse(j)[0].normal_form)
        except:  # если этого не получается сделать пропускаем данное слово
            pass
    buf_of_request = []
    for j in range(len(dict_of_words)):
        buf_of_request.append(0)
    for j in buf4:
        if j in dict_of_words:
            buf_of_request[dict_of_words.get(j)] += 1
    near_vectors = []
    for i in range(0, 130, 1):
        near_vectors.append((df['1'].values[i], distance.cosine(vectors_of_definitions[i][1], buf_of_request)))
    near_vectors.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        result.append(near_vectors[i][0])
    return result


def bert_logical(input_request):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
    df = pd.read_csv('dataframe.csv')  # загружаем частично подготовленный data frame
    definition_list = []
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
    tokens_sen_2 = summarize(input_request, ratio=0.1)
    tokens_sen_2 = tokenizer.tokenize(tokens_sen_2)
    probability_logical = []
    for i in range(0, 130, 1):
        buf = df['2'].values[i]
        definition_list.append(summarize(buf, ratio=0.1))
        tokens_sen_1 = tokenizer.tokenize(definition_list[i])
        tokens = ['[CLS]'] + tokens_sen_1 + ['[SEP]'] + tokens_sen_2 + ['[SEP]']
        token_input = tokenizer.convert_tokens_to_ids(tokens)
        token_input = token_input + [0] * (512 - len(token_input))
        mask_input = [0] * 512
        seg_input = [0] * 512
        len_1 = len(tokens_sen_1) + 2  # длина первой фразы, +2 - включая начальный CLS и разделитель SEP
        for j in range(len(tokens_sen_2) + 1):  # +1, т.к. включая последний SEP
            seg_input[len_1 + j] = 1  # маскируем вторую фразу, включая последний SEP, единицами
        # конвертируем в numpy в форму (1,) -> (1,512)
        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])
        predicts = model.predict([token_input, seg_input, mask_input])[1]
        probability_logical.append([df['1'].values[i], int(round(predicts[0][0] * 100))])
    probability_logical.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        result.append(probability_logical[i][0])
    return result


def bag_of_word_euclidian(input_request):  # DONE
    morph = pymorphy2.MorphAnalyzer()  # подключаем анализатор слов
    df = pd.read_csv('dataframe.csv')  # загружаем частично подготовленный data frame
    all_words = ""
    for i in range(1, 130, 1):
        all_words += df['1'].values[i] + ' ' + df['2'].values[i]
    buf = gensim.utils.simple_preprocess(all_words)  # очищаем запрос от второстепенных данных
    buf2 = ' '.join(buf)  # приводим полученный вид запроса в строку
    buf3 = re.split(r'[ \–\, .]+', buf2)  # удаляем знаки пунктуации
    buf4 = []
    for j in buf3:
        try:
            buf4.append(morph.parse(j)[0].normal_form)
        except:  # если этого не получается сделать пропускаем данное слово
            pass
    dict_of_words = {}
    i = 0
    for j in buf4:
        if not j in dict_of_words:
            dict_of_words.update({j: i})
            i += 1
    vectors_of_definitions = []
    for i in range(0, 130, 1):
        buf = []
        for j in range(len(dict_of_words)):
            buf.append(0)
        definition = df['2'].values[i]
        buf2 = gensim.utils.simple_preprocess(definition)  # очищаем запрос от второстепенных данных
        buf3 = ' '.join(buf2)  # приводим полученный вид запроса в строку
        buf4 = re.split(r'[ \–\, .]+', buf3)
        buf5 = []
        for j in buf4:
            try:
                buf5.append(morph.parse(j)[0].normal_form)
            except:  # если этого не получается сделать пропускаем данное слово
                pass
        for j in buf5:
            if j in dict_of_words:
                buf[dict_of_words.get(j)] += 1
        sum = 0
        for j in buf:
            sum = j ** 2
        sum = sqrt(sum)
        if sum != 0:
            for j in range(len(buf)):
                buf[j] /= sum
        vectors_of_definitions.append([df['1'].values[i], buf])
    buf = gensim.utils.simple_preprocess(input_request)  # очищаем запрос от второстепенных данных
    buf2 = ' '.join(buf)  # приводим полученный вид запроса в строку
    buf3 = re.split(r'[ \–\, .]+', buf2)  # удаляем знаки пунктуации
    buf4 = []
    for j in buf3:
        try:
            buf4.append(morph.parse(j)[0].normal_form)
        except:  # если этого не получается сделать пропускаем данное слово
            pass
    buf_of_request = []
    for j in range(len(dict_of_words)):
        buf_of_request.append(0)
    for j in buf4:
        if j in dict_of_words:
            buf_of_request[dict_of_words.get(j)] += 1
    sum1 = 0
    for j in buf_of_request:
        sum1 += j ** 2
    sum1 = sqrt(sum1)
    if sum1 == 0:
        return 'Не могу ничего подсказать, опишите подробнее'
    for j in range(len(buf_of_request)):
        buf_of_request[j] /= sum1
    near_vectors = []
    for i in range(0, 130, 1):
        near_vectors.append((df['1'].values[i], distance.euclidean(vectors_of_definitions[i][1], buf_of_request)))
    near_vectors.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        result.append(near_vectors[i][0])
    return result


def elmo_embedding():
    df = pd.read_csv('dataframe.csv')  # загружаем частично подготовленный data frame
    vectors_def = []
    for i in range(0, 130, 1):
        vectors_def.append(elmo_vector_create(df['2'].values[i]))
    vectors_def_2 = []
    for i in range(len(vectors_def)):
        vectors_def_2.append(str(vectors_def[i]))
    final_vectors = [vectors_def_2]
    f = open('model_vectors_elmo.txt', 'w')  # открываем файл для записи векторов определений
    json.dump(final_vectors, f)  # записываем json наш список списков векторов
    f.close()  # закрываем файл
