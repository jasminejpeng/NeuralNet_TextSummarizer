import wikipedia
from collections import Counter
import unidecode
import csv
import torch


def create_tensors(filename):
    csvlines = extract_vectors('english_vectors.csv')
    vectorlist = []
    for i in csvlines:
        numlist = []
        if i:
            if i[1]:
                for j in range(len(i[1])):
                    char = i[1][j]
                    assert char != ''
                    if char.isnumeric():
                        numlist.append(float(char))
        wordvector = torch.tensor(numlist)
        wordvector = wordvector.float()
        vectorlist.append(wordvector)
    tensorlist = []
    for t in vectorlist:
        if t.numel():
            tensorlist.append(t)
    return tensorlist


def create_csv(readfile, writefile):
    file = open(readfile, 'r')
    wordlist = file.read()
    file.close()
    wordlist = wordlist.split('\n')
    wordlist.pop()
    vectorlist = []
    for word in wordlist:
        vectorlist.append(vectorize_word(chars2nums(word)))
    rows = list(zip(wordlist, vectorlist))
    csvfile = open(writefile, 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)
    csvfile.close()


def create_txt(filename, lang, chars=500000, words=3000):
    words = compress(list_processor(unidecode.unidecode(wiki_pull(chars, lang)).split()), words)
    file = open(filename, 'w')
    for word in words:
        file.write(word)
        file.write('\n')
    file.close()


def wiki_pull(chars, lang):
    wikipedia.set_lang(lang)
    string = ''
    while len(string) < chars:
        try:
            string += wikipedia.page(wikipedia.random()).content
        except wikipedia.exceptions.DisambiguationError as e:
            string += wikipedia.page(e.options[0]).content
        except wikipedia.exceptions.PageError:
            string += wikipedia.page(wikipedia.random()).content
    return string


def check_similarity(list1, list2):
    count = 0
    for i in range(10000):
        for j in range(10000):
            if list1[i] == list2[j]:
                count += 1
    return count/10000**2


def list_processor(words):
    n = len(words)
    for i in range(n):
        word = words[i]
        word = list([char for char in word if char.isalpha()])
        word = "".join(word)
        word = word.lower()
        words[i] = word
    wordlist = []
    for w in words:
        if 6 >= len(w) > 0:
            wordlist.append(w)
    return wordlist


def chars2nums(word):
    nums = []
    for char in word:
        num = ord(char) - 97
        nums.append(num)
    return nums


def vectorize_word(word):
    wordvect = []
    for n in range(6):
        for i in range(26):
            if len(word) > n:
                if word[n] == i:
                    wordvect.append(1)
                else:
                    wordvect.append(0)
            else:
                wordvect.append(0)
    return wordvect


def extract_vectors(filename):
    csvfile = open(filename, 'r')
    csvreader = csv.reader(csvfile)
    vectorlist = []
    for row in csvreader:
        vectorlist.append(row)
    csvfile.close()
    return vectorlist


def compress(string, n):
    stringcopy = Counter(string)
    tuples = stringcopy.most_common(n)
    words = []
    for i in tuples:
        words.append(i[0])
    return words



