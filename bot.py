from sys import argv
import new_preproc
import pickle
from pymystem3 import Mystem
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


def preprocess_sent(text):
    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    letter = re.compile(r'[А-Яа-я]+')

    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if letter.findall(token)
              and letter.findall(token)[0] == token
              and token not in russian_stopwords]

    text = " ".join(tokens)

    vectors = new_preproc.preproc_texts([text])
    return vectors


if __name__ == '__main__':
    if len(argv) == 1:
        print('Введите предложение:')
        sent = input()
        sent_num = preprocess_sent(sent)

        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(sent_num)
        print(y_pred)

    elif argv[1] == '--help':
        print('Для запуска введите:'
              '     python bot.py')

    else:
        print('Введите python bot.py --help для вызова справки.')
