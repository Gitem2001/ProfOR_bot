import re
from math import sqrt

import gensim
from scipy.spatial import distance
import pandas as pd
from vector_create import vector_create_by_str
from vector_create import elmo_vector_create
import json


def request(input_request):
    df = pd.read_csv('dataframe.csv')  # подключаем dataframe
    f = open('model_vectors.txt', 'r')  # открываем файл с заранее заготовленными векторами определений
    final_vector = vector_create_by_str(input_request)  # создаем вектор запроса
    if final_vector[0] == 'Cant word2vec':  # если не удалось преобразовать запрос в вектор
        return 'Не могу ничего подсказать, опишите подробнее'
    vectors_def = json.load(f)  # загружаем заранее заготовленные векторы определений
    for i in range(len(vectors_def)):
        for j in range(len(vectors_def[i])):
            vectors_def[i][j] = float(vectors_def[i][j])  # парсим json в изначальный формат обьекта
    near_vectors = []  # создаем список для нахождения ближайших векторов
    for i in range(1, 129, 1):  # передвигаемся по списку векторов определений сравнивая с пользовательским запросом
        near_vectors.append(
            (df['1'].values[i], distance.cosine(vectors_def[i], final_vector)))  # вычисляем для каждого косинусное
        # расстояние
    near_vectors.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        print(near_vectors[i][1])
        result.append(near_vectors[i][0])
    return result  # возвращаем список ближайших профессий


def request_elmo(input_request):
    df = pd.read_csv('dataframe.csv')
    f = open('model_vectors_elmo.txt', 'r')
    vector_of_request = elmo_vector_create(input_request)
    if vector_of_request[0] == 'Cant word2vec':  # если не удалось преобразовать запрос в вектор
        return 'Не могу ничего подсказать, опишите подробнее'
    vectors_def = json.load(f)  # загружаем заранее заготовленные векторы определений
    for i in range(len(vectors_def)):
        for j in range(len(vectors_def[i])):
            vectors_def[i][j] = float(vectors_def[i][j])  # парсим json в изначальный формат обьекта
    near_vectors = []
    for i in range(1, 129, 1):
        near_vectors.append((df['1'].values[i], distance.cosine(vectors_def[i], vector_of_request)))
    near_vectors.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        result.append(near_vectors[i][0])
    return result


def request_euclidian(input_request):
    df = pd.read_csv('dataframe.csv')  # подключаем dataframe
    f = open('model_vectors.txt', 'r')  # открываем файл с заранее заготовленными векторами определений
    final_vector = vector_create_by_str(input_request)  # создаем вектор запроса
    if final_vector[0] == 'Cant word2vec':  # если не удалось преобразовать запрос в вектор
        return 'Не могу ничего подсказать, опишите подробнее'
    sum1 = 0
    for i in final_vector:  # передвигаемся по списку векторов определений сравнивая с пользовательским запросом
        sum1 += i**2
    sum1 = sqrt(sum1)
    for i in range(len(final_vector)):
        final_vector[i] /= sum1
    vectors_def = json.load(f)  # загружаем заранее заготовленные векторы определений
    for i in range(len(vectors_def)):
        for j in range(len(vectors_def[i])):
            vectors_def[i][j] = float(vectors_def[i][j])  # парсим json в изначальный формат обьекта
    near_vectors = []  # создаем список для нахождения ближайших векторов
    for i in range(1, 129, 1):  # передвигаемся по списку векторов определений сравнивая с пользовательским запросом
        near_vectors.append((df['1'].values[i], distance.euclidean(vectors_def[i], final_vector)))  # вычисляем для каждого косинусное
        # расстояние
    near_vectors.sort(key=lambda touple: touple[1])  # сортируем по косинусному расстоянию
    result = []  # создаем результирующий список подходящих профессий
    for i in range(0, 5):  # выводим 5 профессий самых ближайших к пользовательскому запросу
        print(near_vectors[i][1])
        result.append(near_vectors[i][0])
    return result  # возвращаем список ближайших профессий
