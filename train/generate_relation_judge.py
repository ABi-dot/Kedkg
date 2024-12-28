import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm
import openai
def link_entity(entity, nlp):
    pattern = r'Q\d+'
    name = entity
    linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
    if linking:
        linking = linking.group(0) # Q?
    else:
        linking = name
    return linking

with open('../output/qid2name_new.pkl', 'rb') as f:
    qid2name = pickle.load(f)
    print("Loading qid2name complete!")

with open('../output/triples_new.pkl', 'rb') as f:
    triples = pickle.load(f)
    print("Loading triples complete!")

with open('../output/relations_new.pkl', 'rb') as f:
    relations = pickle.load(f)
    print("Loading relations complete!")

train = []

with open("../datasets/MQuAKE-CF.json", 'r') as input:
    data = json.load(input)
    nlp = spacy.load("../en_core_web_md") # entity linking model
    nlp.add_pipe("entityLinker", last=True)
    for i in tqdm(range(len(data))):
        d = data[i]
        edits = d["requested_rewrite"]
        for edit in edits:
            q = edit['question']
            head = edit['subject']
            head_linking = link_entity(head, nlp)
            tmp_rels = relations[head_linking]
            # print((head, head_linking, tmp_rels))
            if len(tmp_rels) == 0:
                continue
            # find rels
            for rel in tmp_rels:
                t = dict()
                t['question'] = q
                t['relation'] = rel
                # get all posible tail entitys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                tail_linkings = list(triples[(head_linking, rel)])
                for tail_linking in tail_linkings:
                    tail_linking = tail_linking[0] #get entity
                    tail = qid2name[tail_linking]
                    if edit["target_new"]["str"] in tail:
                        t['label'] = 1
                        break
                else:
                    t['label'] = 0
                train.append(t)

with open('../train/datasets_relation_judge.json', 'w') as output:
    json.dump(train, output, indent=4)
            