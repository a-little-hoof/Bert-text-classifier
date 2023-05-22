import re
import unicodedata
import jieba
import random
import torch

SOS_token = 0
EOS_token = 1

class Lang:
    
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2
    
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def addSentence_cn(self,sentence):
        for word in list(jieba.cut(sentence)):
            self.addWord(word)

def readLangs():
    print("Reading lines...")
    lines = open('./cmn-eng.txt',encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang("cmn")
    output_lang = Lang("eng")
    
    return input_lang,output_lang,pairs

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn')

###去空格，非字母符号
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1",s)
    return s

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < 20 and len(p[1].split(' ')) < 20 and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData():
    input_lang,output_lang,pairs = readLangs()
    print("Read %s sentence pairs"%len(pairs))
    print(random.choices(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs"%len(pairs))
    for pair in pairs:
        input_lang.addSentence_cn(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name,input_lang.n_words)
    print(output_lang.name,output_lang.n_words)
    return input_lang,output_lang,pairs


def tensorFromSentence(lang,sentence,config):
    if lang.name == 'cmn':
        indexes = [lang.word2index[word] for word in list(jieba.cut(sentence))]
    if lang.name == 'eng':
        indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    ###列表转化为n*1向量
    return torch.tensor(indexes, dtype=torch.long, device=config.device).view(-1, 1)


def tensorsFromPair(input_lang,output_lang,pair,config):
    input_tensor = tensorFromSentence(input_lang, pair[0],config)
    target_tensor = tensorFromSentence(output_lang, pair[1],config)
    return (input_tensor, target_tensor)

def main():
    input_lang,output_lang,pairs = prepareData()
    print(random.choices(pairs))

if __name__ == "__main__":
    main()