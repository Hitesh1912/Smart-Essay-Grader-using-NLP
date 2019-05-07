import os
import glob
import nltk
import string
from nltk.corpus import stopwords
import re


def write_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8-sig', mode='w') as fp:
        for character, count in data.items():
            fp.write('{},{}\n'.format(character, count))


def write_dict_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8-sig', mode='w') as fp:
        for character, count in data:
            fp.write('{},{}\n'.format(character, count))


def normal_write_to_file(filename, data):
    target_dir = r"Results/"
    filename = target_dir + filename + ".txt"
    with open(filename, encoding='utf-8', mode='w') as fp:
        for item in data:
            fp.write('{}\n'.format(item))


def fileToCollection(filename):
    contents = open(filename, 'r', encoding='utf-8')
    collection = contents.readlines()
    collection_dict ={}
    for line in collection:
        if line.strip() == '': continue
        line = line.strip().split()
        # print (line)
        gram = line[0]
        # print (qId)
        freq= line[2]
        if gram !='Character':
            if gram not in collection_dict.keys():
                collection_dict[gram] = [freq]
            else:
                collection_dict[gram].append(freq)
    # print (collection_dict)
    return collection_dict


def combine_files(filename):
    result_filename = filename
    # read_files = glob.glob("gutenberg/*.txt")
    read_files = glob.glob("brown/*.txt")
    with open(result_filename, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())


def rename_files(dir):
    filelist = os.listdir(os.path.abspath(dir))
    ext = '.txt'
    print(filelist)
    for fname in filelist:
        os.rename(dir+fname, dir+fname + ext)


def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Stemming
    text = text.split()
    stemmer = nltk.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text





