import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm
import openai
tokenizer = AutoTokenizer.from_pretrained("../model/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../model/rebel-large")
DEVICE = "cuda:1"
def link_entity(entity, nlp):
    pattern = r'Q\d+'
    name = entity
    linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
    if linking:
        linking = linking.group(0) # Q?
    else:
        linking = name
    return linking

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
                            return_tensors='pt')
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


train = []

with open("../datasets/MQuAKE-CF.json", 'r') as input:
    data = json.load(input)
    nlp = spacy.load("../en_core_web_md") # entity linking model
    nlp.add_pipe("entityLinker", last=True)
    for i in tqdm(range(len(data[:]))):
        d = data[i]
        edits = d["requested_rewrite"]
        for edit in edits:
            q = edit['question']
            kb = from_small_text_to_kb(q, verbose=True)
            # print(kb.relations)
            kb = kb.relations
            entity_set = set()
            for k in kb:
                entity_set.add(k['head'])
                entity_set.add(k['type'])
                entity_set.add(k['tail'])
            for entity in entity_set:
                t = dict()
                t['question'] = q
                t['entity'] = entity
                if entity == edit['subject']:
                    t['label'] = 1
                else:
                    t['label'] = 0
                train.append(t)

with open('../train/datasets_entity_judge.json', 'w') as output:
    json.dump(train, output, indent=4)
            