import re
from math import sqrt

import gensim
import gensim.downloader as api
import pymorphy2
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def vector_create_by_str(input_str):
    model = api.load("word2vec-ruscorpora-300")  # подключаем модель word2vec с русским словарем
    morph = pymorphy2.MorphAnalyzer()  # подключаем анализатор слов
    buf = gensim.utils.simple_preprocess(input_str)  # очищаем запрос от второстепенных данных
    buf2 = ' '.join(buf)  # приводим полученный вид запроса в строку
    buf3 = re.split(r'[ \–\, .]+', buf2)  # удаляем знаки пунктуации
    buf4 = []  # создаем список для векторов слов запроса
    for j in buf3:  # передвигаемся по новому очищенному запросу
        try:
            buf4.append(model[morph.parse(j)[0].normal_form + '_' + morph.parse(j)[
                0].tag.POS])  # пробуем перевести слово в вектор
        except:  # если этого не получается сделать пропускаем данное слово
            pass
    if len(buf4) == 0:  # если преобразовать ниодно слово не удалось возвращаем строку невозможности
        return ['Cant word2vec']
    final_vector = buf4[0]  # инициализируем переменную вектора запроса вектором первого слова
    for j in range(1, len(buf4), 1):  # передвигаемся по списку векторов слов
        final_vector = [final_vector + buf4[j] for final_vector, buf4[j] in
                        zip(final_vector, buf4[j])]  # суммируем векторы слов запроса получая вектор запроса
    sum = 0
    for j in range(len(final_vector)):
        sum += final_vector[j] ** 2
    sum = sqrt(sum)
    if sum == 0:
        return ['Cant word2vec']
    for i in range(len(final_vector)):
        final_vector[i] /= sum
    return final_vector  # возвращаем вектор запроса


def elmo_vector_create(input_str):
    elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                      trainable=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    buf = gensim.utils.simple_preprocess(input_str)
    buf2 = ' '.join(buf)
    buf3 = re.split(r'[ \–\, .]+', buf2)
    if len(buf3) == 0:  # если преобразовать ниодно слово не удалось возвращаем строку невозможности
        return ['Cant word2vec']
    embeddings = elmo(buf3, signature="default", as_dict=True)["elmo"]
    vectors_of_word = sess.run(embeddings)
    vector_of_def = vectors_of_word[0]
    for j in range(1, len(vectors_of_word), 1):  # передвигаемся по списку векторов слов
        vector_of_def = [vector_of_def + vectors_of_word[j] for vector_of_def, vectors_of_word[j] in
                         zip(vector_of_def,
                             vectors_of_word[j])]  # суммируем векторы слов запроса получая вектор запроса

    return vector_of_def
