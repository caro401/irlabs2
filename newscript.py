import nltk
import numpy
import os
from collections import Counter

def extract_place_names(t):   # TODO modify this
    entity_names =[]

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_place_names(child))

    return entity_names


files = [f for f in os.listdir(".") if f.endswith('.txt')]  # find all the files in the current working directory with extension .txt

for file in files:
    print(file)
    with open(file, encoding='utf-8') as f:  
        sample = f.read()
    
    
    sentences = nltk.sent_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)  # this is a generator!
    
    
    place_names = []
    for tree in chunked_sentences:
        # Print results per sentence
        # print extract_entity_names(tree)
    
        place_names.extend(extract_place_names(tree))
    
    print(Counter(place_names))
