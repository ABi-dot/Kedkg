from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import re
import json
import spacy
import pickle
import collections
from tqdm import tqdm
# Load model and tokenizer
DEVICE = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained("../model/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../model/rebel-large").to(DEVICE)


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")
    
def from_small_text_to_kb(text, verbose=False):
    kb = KB()

    # Tokenizer text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                            return_tensors='pt').to(DEVICE)
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

    # Generate
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 3
    }
    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb


qid2name = collections.defaultdict(str)
triples = collections.defaultdict(set)
qid_relations = collections.defaultdict(set)
with open("../datasets/MQuAKE-CF.json", 'r') as input:
    data = json.load(input)
    for i in tqdm(range(len(data))):
        d = data[i]
        edits = d["requested_rewrite"]
        for edit in edits:
            sentence = edit["prompt"].format(edit["subject"]) + ' ' + edit["target_new"]["str"]
            text = sentence
            kb = from_small_text_to_kb(text, verbose=True)
            print(kb.relations)
            nlp = spacy.load("../en_core_web_md")
            nlp.add_pipe("entityLinker", last=True)
            pattern = r'Q\d+'
            for triple in kb.relations:
                head = triple['head']
                tail = triple['tail']
                rel = triple['type']
                if edit["target_true"]["str"] in head or edit["target_true"]["str"] in tail:
                    continue
                
                head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))
                tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
                if head_linking:
                    head_linking = head_linking.group(0) # Q?
                else:
                    head_linking = head
                
                print(head_linking)
                
                if head_linking not in qid2name:
                    qid2name[head_linking] = head
                
                
                if tail_linking:
                    tail_linking = tail_linking.group(0)
                else:
                    tail_linking = tail

                if tail_linking not in qid2name:
                    qid2name[tail_linking] = tail

                triples[(head_linking, rel)].add((tail_linking, sentence))
                qid_relations[head_linking].add(rel)



with open('../output/qid2name_new.pkl', 'wb') as f:
    pickle.dump(qid2name, f)

with open('../output/triples_new.pkl', 'wb') as f:
    pickle.dump(triples, f)

with open('../output/relations_new.pkl', 'wb') as f:
    pickle.dump(qid_relations, f)