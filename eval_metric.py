import re
import os
import argparse
import json
import random
import torch
import bert_score
from transformers import (
    AutoModel, 
    AutoTokenizer,
    DebertaModel,  # Changed from DebertaV3Model
    DebertaTokenizer,  # Changed from DebertaV3Tokenizer
    BartModel,
    BartTokenizer,
    T5Model,
    T5Tokenizer
)
from fbd_score import *
from prd_score import *
from baseline import cal_bleu, cal_meteor, cal_rouge, cal_greedy_match, cal_embd_average, cal_vec_extr
from pce import calculate_information_scores, volume_of_unit_ball_log, cross_entropy, entropy
import math
from scipy.stats import spearmanr, pearsonr


parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='dialogue', help='[dialogue | mt]')
parser.add_argument('--data_path', type=str, help='path to dialogue annotation data')
parser.add_argument('--src_path', type=str, help='path to MT sources')
parser.add_argument('--ref_path', type=str, help='path to MT references')
parser.add_argument('--hyp_path', type=str, help='path to MT hypotheses')
parser.add_argument('--human_path', type=str, help='path to human annotations')
parser.add_argument('--metric', type=str, help='[bleu | meteor | rouge | greedy | average | extrema | bert_score | fbd | prd]')
parser.add_argument('--sample_num', type=int, default=10, help='sample number of references')
parser.add_argument('--model_type', type=str, default='', help='pretrained model type or path to pretrained model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--is_chinese', type=int, default=0, help='Is Chinese corpus or not')
parser.add_argument('--embedding_model', type=str, default='roberta', 
                   choices=['roberta', 'deberta', 'bart', 't5'], 
                   help='Type of embedding model to use')
args = parser.parse_args()

def get_modern_model_configs(model_type, embedding_model, is_chinese):
    if embedding_model == 'deberta':
        model_name = "microsoft/deberta-base"  # Changed to base model
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        model = DebertaModel.from_pretrained(model_name)
    elif embedding_model == 'bart':
        model_name = "facebook/bart-large"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartModel.from_pretrained(model_name)
    elif embedding_model == 't5':
        model_name = "t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5Model.from_pretrained(model_name)
    else:  # default to roberta
        return get_model_configs(model_type, is_chinese)
    
    return tokenizer, model

def get_modern_embeddings(querys, answers, tokenizer, model, batch_size, use_cuda=True, embedding_model='roberta'):
    if embedding_model == 'roberta':
        return get_embeddings(querys, answers, tokenizer, model, batch_size, use_cuda)
    
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(querys), batch_size):
        batch_querys = querys[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]
        
        texts = [q + tokenizer.sep_token + a for q, a in zip(batch_querys, batch_answers)]
        
        inputs = tokenizer(texts, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True, 
                         max_length=512).to(device)
        
        with torch.no_grad():
            if embedding_model == 'deberta':
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif embedding_model == 'bart':
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.encoder_last_hidden_state.mean(dim=1)
            elif embedding_model == 't5':
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
        all_embeddings.append(embeddings.cpu())  # Move to CPU here
        
    return torch.cat(all_embeddings, dim=0).numpy() 
    
def read_mt_data(args):
    querys, refs, hyps, human_scores = [], [], [], []

    with open(args.src_path, 'r', encoding='utf-8') as f:
        for line in f:
            querys.append(line.strip())
    with open(args.ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            refs.append([line.strip()])

    files = os.listdir(args.hyp_path)
    system_list = []
    for file_ in files:
        hyps.append([])
        system = re.findall(r'news\w*\.(.*)\.\w{2}\-\w{2}', file_)[0]
        system_list.append(system)
        with open(os.path.join(args.hyp_path, file_), 'r', encoding='utf-8') as f:
            for line in f:
                hyps[-1].append(line.strip())
                
    human_scores = [[0 for _ in range(len(system_list))]]
    with open(args.human_path, 'r', encoding='utf-8') as f:
        for line in f:
            system, score = line.split()
            human_scores[0][system_list.index(system)] = float(score)

    return querys, refs, hyps, human_scores
    

def read_dialogue_data(path):
    querys = []
    refs = []
    hyps = []
    human_scores = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            querys.append(line['src'])
            refs.append(line['refs'])

            for i, hyp in enumerate(line['hyps']):
                if len(hyps) < i + 1:
                    hyps.append([])
                hyps[i].append(hyp)

            for i, scores in enumerate(line['human_scores']):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                for j, score in enumerate(scores):
                    if len(human_scores[i]) < j + 1:
                        human_scores[i].append([])
                    human_scores[i][j].append(score)
    
    return querys, refs, hyps, human_scores

def sample(lists, num):
    for i in range(len(lists)):
        if num < len(lists[i]):
            lists[i] = random.sample(lists[i], num)
    return lists

def average(lists):
    for i in range(len(lists)):
        lists[i] = [sum(lst) / len(lst) for lst in lists[i]]
    return lists

def prepare_data(querys, refs):
    target_querys, target_answers = [], []
    for query, answers in zip(querys, refs):
        for answer in answers:
            target_querys.append(query)
            target_answers.append(answer)
    return target_querys, target_answers

def eval_metric(args):
    if args.task_type == 'dialogue':
        querys, refs, hyps, human_scores = read_dialogue_data(args.data_path)
        average_human_scores = average(human_scores)
        human_scores = []
        for scores in average_human_scores:
            for i, score in enumerate(scores):
                if len(human_scores) < i + 1:
                    human_scores.append([])
                human_scores[i].append(score)
    else:
        querys, refs, hyps, human_scores = read_mt_data(args)

    refs = sample(refs, args.sample_num)

    system_scores = []
    print("#-------------------------------------#")
    print(args.metric, args.model_type)
    print("#-------------------------------------#")
    assert args.metric in ['rouge', 'meteor', 'greedy', 'average', 'extrema', 'bert_score', 'fbd', 'prd', 'bleu', 'itm']
    if args.metric == 'bert_score':
        for hyp in hyps:
            score = bert_score.score(hyp, refs, model_type=args.model_type, batch_size=args.batch_size)
            score = score[2].mean(dim=0).cpu().item()
            system_scores.append(score)

    elif args.metric == 'bleu':
        for hyp in hyps:
            system_scores.append(cal_bleu(refs, hyp, args.is_chinese))

    elif args.metric == 'meteor':
        for hyp in hyps:
            system_scores.append(cal_meteor(refs, hyp))

    elif args.metric == 'rouge':
        for hyp in hyps:
            system_scores.append(cal_rouge(refs, hyp))

    elif args.metric == 'greedy':
        system_scores = cal_greedy_match(refs, hyps)

    elif args.metric == 'average':
        system_scores = cal_embd_average(refs, hyps)

    elif args.metric == 'extrema':
        system_scores = cal_vec_extr(refs, hyps)

    else:
        source_querys = querys
        source_answer_list = hyps
        target_querys, target_answers = prepare_data(querys, refs)
        tokenizer, model = get_modern_model_configs(args.model_type, args.embedding_model, args.is_chinese)

        if args.metric == 'fbd':
            mu1, sigma1 = get_statistics(target_querys, target_answers, tokenizer, 
                                       model, args.batch_size, use_cuda=True,
                                       embedding_model=args.embedding_model)
            for source_answers in source_answer_list:
                mu2, sigma2 = get_statistics(source_querys, source_answers, tokenizer, 
                                           model, args.batch_size, use_cuda=True,
                                           embedding_model=args.embedding_model)
                score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
                system_scores.append(score)
                
        elif args.metric in ['prd', 'itm']:
            reference_feats = get_modern_embeddings(target_querys, target_answers, tokenizer, 
                                                  model, args.batch_size, use_cuda=True,
                                                  embedding_model=args.embedding_model)
            
            for source_answers in source_answer_list:
                system_feats = get_modern_embeddings(source_querys, source_answers, tokenizer, 
                                                   model, args.batch_size, use_cuda=True,
                                                   embedding_model=args.embedding_model)
                
                if args.metric == 'prd':
                    precision, recall = compute_prd_from_embedding(system_feats, reference_feats, 
                                                                 enforce_balance=False)
                    precision = precision.tolist()
                    recall = recall.tolist()
                    max_f1_score = max([2*p*r/(p+r + 1e-6) for p,r in zip(precision, recall)])
                    system_scores.append(max_f1_score)
                else:  # itm
                    scores = calculate_information_scores(
                        source_embeddings=system_feats,  # Now it's numpy array
                        target_embeddings=reference_feats,  # Now it's numpy array
                        k=2, C=3
                    )
                    system_scores.append(scores['cross_entropy_ts'])

        else:
            raise NotImplementedError(f"We don't support the metric: {args.metric}")
            
    pearson_corrs = []
    spearman_corrs = []
    print("SCORES")
    for scores in human_scores:
        print(system_scores, scores)
        pearson_corrs.append(abs(pearsonr(system_scores, scores)[0]))
        spearman_corrs.append(abs(spearmanr(system_scores, scores)[0]))
    print('The pearson correlation between {} and human score is {}'.format(args.metric, pearson_corrs))
    print('The spearman correlation between {} and human score is {}'.format(args.metric, spearman_corrs))

if __name__ == '__main__':
    eval_metric(args)
