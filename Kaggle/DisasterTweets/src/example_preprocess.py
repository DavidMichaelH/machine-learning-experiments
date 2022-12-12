from clean_and_preprocess import CleanAndPreprocess
import pandas as pd

# ------ pandas ---------


labeled = pd.pandas.read_csv("data/labeled_data.csv")

data_cleaner = CleanAndPreprocess(dataframe=labeled)

print(data_cleaner.dataframe)

# ---- Replacing tweet specific elements -----
replacements = []
replacements.append((lambda x: x.startswith('@'), '@user'))
replacements.append((lambda x: x.startswith('http'), 'http'))
replacements.append((lambda x: x.startswith('#'), '#hashtag'))
replacements.append((lambda x: x.isnumeric(), 'number'))

data_cleaner.replace_in_rows(columns=['text'], replacements=replacements, delimiter=' ')

import string

exclude = list(string.punctuation)
exclude.remove('@')  # twitter specific
exclude.remove('#')  # twitter specific

data_cleaner.remove_from_rows(columns=['text'], removals=exclude, delimiter='')

data_cleaner.lower_case(columns=['text'])


print(data_cleaner.dataframe.head(10))

import nltk
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].lower()
    tag_dict = {"a": wordnet.ADJ,
                "n": wordnet.NOUN,
                "v": wordnet.VERB,
                "r": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)




def lemmatize_sentence(sent):
    sent_list = []
    for w in sent.split(" "):
        lem_w = lemmatizer.lemmatize(w, pos=get_wordnet_pos(w))
        sent_list.append(lem_w)


    sent = " ".join(sent_list)
    return sent


replacements = [(lambda x: True, lambda w: lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)))]

data_cleaner.replace_in_rows(columns=['text'], replacements=replacements, delimiter=' ')

print(data_cleaner.dataframe)