import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm
import openai

train = []

with open("../datasets/MQuAKE-CF.json", 'r') as input:
    data = json.load(input)
    # nlp = spacy.load("../en_core_web_md") # entity linking model
    # nlp.add_pipe("entityLinker", last=True)
    for i in tqdm(range(len(data[:]))):
        d = data[i]
        for q in d['questions']:
            t = dict()
            t['question'] = q
            t['subquestions'] = []
            t['subquestions'].append(d["new_single_hops"][0]["question"])
            nxt_entity = d["new_single_hops"][0]["answer"]
            for i in range(1, len(d["new_single_hops"])):
                t['subquestions'].append(d["new_single_hops"][i]["question"].replace(nxt_entity, '[ENT]'))
                nxt_entity = d["new_single_hops"][i]["answer"]
            train.append(t)

with open('../train/datasets_divide.json', 'w') as output:
    json.dump(train, output, indent=4)
            