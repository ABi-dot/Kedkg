import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import random
from sentence_transformers import SentenceTransformer, util
import re
import openai
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import collections
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--edit', type=int, default=1)
args = parser.parse_args()


DEVICE = "cuda:0"
ENTITY = 0
REL = 1
tokenizer = AutoTokenizer.from_pretrained("../model/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../model/rebel-large")
model.to(DEVICE)
# semantic_model = SentenceTransformer("/data/public/model/multi-qa-mpnet-base-dot-v1").to(DEVICE)
TRAIN_PATH = '../train/results/best_model'
ENTITY_TRAIN_PATH = '../train/results_entity_judge/best_model_entity_judge'
bert_tokenizer = DistilBertTokenizer.from_pretrained(TRAIN_PATH)
bert_model = DistilBertForSequenceClassification.from_pretrained(TRAIN_PATH, num_labels=2)
bert_model.to(DEVICE)



entity_bert_tokenizer = DistilBertTokenizer.from_pretrained(ENTITY_TRAIN_PATH)
entity_bert_model = DistilBertForSequenceClassification.from_pretrained(ENTITY_TRAIN_PATH, num_labels=2)
entity_bert_model.to(DEVICE)


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
    # if verbose:
    #     print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

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

def link_entity(entity, nlp):
    pattern = r'Q\d+'
    name = entity
    try:
        linking = re.search(pattern, str(nlp(entity.capitalize())._.linkedEntities))
    except:
        linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
    if linking:
        linking = linking.group(0) # Q?
    else:
        linking = name
    return linking


# def run_llm_answer(query):
#     openai.api_key = "None"
#     openai.api_base = "http://localhost:11451/v1"
#     messages = [{"role":"system","content":"You are an AI assistant that helps people find information. You must provide a short and accurate answer in one sentence. Do not repeat the question. The format is 'Answer:'."}]
#     message_prompt = {"role":"user","content":query}
#     messages.append(message_prompt)
#     f = 0
#     while(f == 0):
        
#         try:
#             response = openai.ChatCompletion.create(
#                     model= openai.Model.list()["data"][0]["id"],
#                     # model = "gpt-3.5-turbo",
#                     messages = messages,
#                     # max_tokens=max_tokens,
#                     frequency_penalty=0,
#                     presence_penalty=0)
#             result = response["choices"][0]['message']['content']
#             f = 1
#         except:
#             print("openai error, retry")
#             time.sleep(10)
#     return result

def run_llm_answer(query):
    openai.api_key = ""
    openai.api_base = "https://api.openai.com/v1"
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information. You must provide a short and accurate answer in one sentence. Do not repeat the question. The format is 'Answer:'."}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        
        try:
            response = openai.ChatCompletion.create(
                    # model= openai.Model.list()["data"][0]["id"],
                    model = "gpt-3.5-turbo",
                    messages = messages,
                    # max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response["choices"][0]['message']['content']
            f = 1
        except:
            print("openai error, retry")
            time.sleep(10)
    return result

def predict(question, relation, FLAG):
    if FLAG == ENTITY:
        inputs = entity_bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
        outputs = entity_bert_model(**inputs)
    elif FLAG == REL:
        inputs = bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
        outputs = bert_model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class = logits.argmax().item()
    return predicted_class, logits, probabilities


def get_scores(question, objects, FLAG):
    scores = []
    for object in objects:
        predicted_class, logits, probabilities = predict(question, object, FLAG)
        score = probabilities[0, 1].item()
        if score < 0.5:
            continue
        scores.append((score, object))
    return scores

def retrieve_from_graph(q):
    kb = from_small_text_to_kb(q, verbose=True)
    print(kb.relations)
    kb = kb.relations
    # choose entity

    entity_set = set()
    for k in kb:
        # capitalize
        entity_set.add(k['head'])
        entity_set.add(k['type'])
        entity_set.add(k['tail'])
    if len(entity_set) == 0:
        return None
    scores = get_scores(q, entity_set, ENTITY)
    if len(scores) == 0:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    print(q, scores)
    head = scores[0][1]
    # choose rels

    head_linking = link_entity(head, nlp)
    tmp_rels = qid_relations[head_linking]
    if len(tmp_rels) == 0:
        return None
    scores = get_scores(q, tmp_rels, REL)
    if len(scores) == 0:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    print(q, scores)
    rel = scores[0][1]
    # find rels
    if rel in tmp_rels:
        # get all posible tail entitys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        tail_linkings = list(triples[(head_linking, rel)])
        # return tail_linkings[0][1] # return sentence
        return qid2name[tail_linkings[0][0]] # return entity
    return None

def verify_gold_path(entities, hops, path):
    if len(entities) != hops:
        return False
    for i in range(hops):
        if entities[i].lower() == path[i]["answer"].lower() or path[i]["answer"].lower() in entities[i].lower() or any(entities[i].lower() in p.lower() for p in path[i]["answer_alias"]):
            continue
        else:
            return False
    return True

def run_llm_divide(query):
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:7000/v1"
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            response = openai.ChatCompletion.create(
                    model= openai.Model.list()["data"][0]["id"],
                    messages = messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response["choices"][0]['message']['content']
            f = 1
        except:
            print("openai error, retry")
            time.sleep(10)
    return result

def construct_graph(batch, qid2name, triples, qid_relations, nlp):
    for i in tqdm(range(len(batch))):
        d = batch[i]
        edits = d["requested_rewrite"]
        for edit in edits:
            # capitalize
            sentence = edit["prompt"].format(edit["subject"]) + ' ' + edit["target_new"]["str"]
            text = sentence
            kb = from_small_text_to_kb(text, verbose=True)
            # print(kb.relations)
            pattern = r'Q\d+'
            for triple in kb.relations:
                # capitalize
                head = triple['head']
                tail = triple['tail']
                rel = triple['type']
                if head == None or head == "" or tail == None or tail == "" or rel == None or rel == "":
                    continue

                # erase true answer
                if edit["target_true"]["str"] in head or edit["target_true"]["str"] in tail:
                    continue
                
                try:
                    head_linking = re.search(pattern, str(nlp(head.capitalize())._.linkedEntities))
                except:
                    head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))

                try:
                    tail_linking = re.search(pattern, str(nlp(tail.capitalize())._.linkedEntities))
                except:
                    tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
                if head_linking:
                    head_linking = head_linking.group(0) # Q?
                else:
                    head_linking = head
                
                # print(head_linking)
                
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

                

def modify_graph(d, qid2name, triples, qid_relations, nlp):
    edits = d["requested_rewrite"]
    for edit in edits:
        sentence = edit["prompt"].format(edit["subject"]) + ' ' + edit["target_new"]["str"]
        text = sentence
        kb = from_small_text_to_kb(text, verbose=True)
        pattern = r'Q\d+'

        clear_flag = collections.defaultdict(int)
        for triple in kb.relations:
            head = triple['head']
            tail = triple['tail']
            rel = triple['type']
            if head == None or head == "" or tail == None or tail == "" or rel == None or rel == "":
                continue

            # erase true answer
            if edit["target_true"]["str"] in head or edit["target_true"]["str"] in tail:
                continue
            
            try:
                head_linking = re.search(pattern, str(nlp(head.capitalize())._.linkedEntities))
            except:
                head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))

            try:
                tail_linking = re.search(pattern, str(nlp(tail.capitalize())._.linkedEntities))
            except:
                tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
            if head_linking:
                head_linking = head_linking.group(0) # Q?
            else:
                head_linking = head
            
            # print(head_linking)
            
            if head_linking not in qid2name:
                qid2name[head_linking] = head
            
            
            if tail_linking:
                tail_linking = tail_linking.group(0)
            else:
                tail_linking = tail

            if tail_linking not in qid2name:
                qid2name[tail_linking] = tail
            
            # remove origin rel
            if clear_flag[(head_linking, rel)] == 0:
                triples[(head_linking, rel)].clear()
                # set flag
                clear_flag[(head_linking, rel)] = 1
                
            
            triples[(head_linking, rel)].add((tail_linking, sentence))
            qid_relations[head_linking].add(rel)
            


with open('../prompts/answer.txt', 'r') as p:
    answer_prompt = p.read()

with open('../prompts/divide.txt','r') as f:
    divide_prompt = f.read()





total = correct = 0
cor = tot = 0
ver_cor = 0
cor_list = [0,0,0]
ver_cor_list = [0,0,0]
tot_list = [0,0,0]
with open("../datasets/MQuAKE-T.json", 'r') as input:
    data = json.load(input)

random.shuffle(data)

batch_edits = args.edit
batch_size = 1868 // batch_edits
batch_data = []
idx = 0
for _ in range(batch_size):
    b = []
    for i in range(batch_edits):
        if idx < len(data):
            b.append(data[idx])
        idx += 1
    batch_data.append(b)
nlp = spacy.load("../en_core_web_md") # entity linking model
nlp.add_pipe("entityLinker", last=True)

for batch in batch_data:
    qid2name = collections.defaultdict(str)
    triples = collections.defaultdict(set)
    qid_relations = collections.defaultdict(set)
    # construct knowledge graph
    construct_graph(batch, qid2name, triples, qid_relations, nlp)

    for i in tqdm(range(len(batch))):
        d = batch[i]
        tot += 1
        tot_list[len(d["single_hops"]) - 2] += 1
        # modify knowledge graph
        modify_graph(d, qid2name, triples, qid_relations, nlp)

        found_ans = gold_path = False
        for q in d['questions']:
            entity_list = []
            # divide into several question
            prompt = divide_prompt.replace("<<<<QUESTION>>>>", q)

            output = run_llm_divide(prompt)
            sub_questions = output.split('\n')
            print("subquesions: " +  str(sub_questions))

            entity = ""
            for subq in sub_questions:
                # replace with entity
                subq = subq.replace("[ENT]", entity)
                retrieve = retrieve_from_graph(subq)
                if retrieve is None:
                    prompt = answer_prompt.replace("<<<<QUESTION>>>>", subq)
                    # not found, pass to LLM
                    output = run_llm_answer(prompt)
                    
                    match = re.search(r'Answer: (.*)', output)
                    if match:
                        entity = match.group(1)
                    else:
                        entity = output
                    print("Not found in graph: " + entity)
                else:
                    entity = retrieve
                    print("Found: " + entity)
                entity_list.append(entity)
            ans = entity

                
            # if the answer is correct -> positive instance for Acc
            if ans.lower() == d["new_answer"].lower() or d["new_answer"].lower() in ans.lower() or any(ans.lower() in k.lower() for k in d["new_answer_alias"]):
                if not found_ans:
                    cor += 1
                    cor_list[len(d["new_single_hops"]) - 2] += 1
                    found_ans = True

                
                gold_path = verify_gold_path(entity_list, len(d["single_hops"]), d["new_single_hops"])

                if gold_path:
                    ver_cor += 1
                    ver_cor_list[len(d["single_hops"]) - 2] += 1
                    break

                
        # print(f"Total: {total}, Correct: {correct},  Total: {correct / total}")
        print(f'Acc = {cor / tot} ({cor} / {tot})')
        print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')
        print(f'2-hop-tot = {tot_list[0]} 2-hop-cor = {cor_list[0]} 2-hop-acc = {ver_cor_list[0]}')
        print(f'3-hop-tot = {tot_list[1]} 3-hop-cor = {cor_list[1]} 3-hop-acc = {ver_cor_list[1]}')
        print(f'4-hop-tot = {tot_list[2]} 4-hop-cor = {cor_list[2]} 4-hop-acc = {ver_cor_list[2]}')
       

            
