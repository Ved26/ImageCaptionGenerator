import re
import json
import collections
import numpy as np

flickr_path = './datasets/Flickr_Data/'
with open(flickr_path+"Flickr_TextData/Flickr8k.token.txt") as filepath:
    captions = filepath.read()
    filepath.close()
captions = captions.split("\n")[:-1]
descriptions = {}
for ele in captions:
    i_to_c = ele.split("\t")
    img_name = i_to_c[0].split(".")[0]
    cap = i_to_c[1]
    if descriptions.get(img_name) == None:
        descriptions[img_name] = []
    descriptions[img_name].append(cap)

def clean_text(sample):
    sample = sample.lower()
    sample = re.sub("[^a-z]+"," ",sample)
    sample = sample.split()
    sample = [s for s in sample if len(s)>1]
    sample = " ".join(sample)
    return sample

for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc_list[i] = clean_text(desc_list[i])
    
f = open("descriptions.txt",'w')
f.write(str(descriptions))
f.close()

f = open("descriptions.txt",'r')
descriptions = f.read()
f.close()

json_acceptable_stirng = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_stirng)

vocabulary = set()
for key in descriptions.keys():
  [vocabulary.update(i.split()) for i in descriptions[key]]


all_vocab = []

for key in descriptions.keys():
  [all_vocab.append(i) for des in descriptions[key] for i in des.split()]


import collections

counter = collections.Counter(all_vocab)
dic_ = dict(counter)
threshold_value = 10
sorted_dic = sorted(dic_.items(),reverse=True,key=lambda x:x[1])
sorted_dic = [x for x in sorted_dic if x[1]>threshold_value]
all_vocab = [x[0] for x in sorted_dic]

f = open(flickr_path+"Flickr_TextData/Flickr_8k.trainImages.txt")
train = f.read()
f.close()
train = [e.split(".")[0] for e in train.split("\n")[:-1]]
f = open(flickr_path+"Flickr_TextData/Flickr_8k.testImages.txt")
test = f.read()
f.close()
test = [e.split(".")[0] for e in test.split("\n")[:-1]]

train_descriptions = {}
for t in train:
    train_descriptions[t]  = []
    for cap in descriptions[t]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[t].append(cap_to_append)

i = 1
word_to_idx = {}
idx_to_word = {}

for word in all_vocab:
    word_to_idx[word] = i
    idx_to_word[i] = word
    i += 1

index = len(word_to_idx)
word_to_idx['startseq'] = index+1
word_to_idx['endseq'] = index+2

idx_to_word[index+1] = 'startseq'
idx_to_word[index+2] = 'endseq'

vocab_size = len(idx_to_word) + 1

all_captions_len = []

for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        all_captions_len.append(len(cap.split()))

max_len = max(all_captions_len)
golve_path ='./datasets/glove.6B.200d.txt'

def get_embeddings_index(glove_path):
    embeddings_index = {}
    glove = open(golve_path, 'r', encoding = 'utf-8').read()
    for line in glove.split("\n"):
        values = line.split(" ")
        word = values[0]
        indices = np.asarray(values[1: ], dtype = 'float32')
        embeddings_index[word] = indices
    return embeddings_index

def get_embedding_output(word_to_idx,embeddings_index):
    emb_dim = 200
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, i in word_to_idx.items():
        emb_vec = embeddings_index.get(word)
        if emb_vec is not None:
            emb_matrix[i] = emb_vec
    return emb_matrix

emb_matrix = get_embedding_output(word_to_idx,get_embeddings_index(golve_path))
