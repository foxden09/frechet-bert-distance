import json
import numpy as np

def read_fed_data(path):
    """
    Read FED format data and convert to our format.
    FED has 18 attributes: 
    - engaging, interesting, uses_knowledge, inquisitive, 
    - consistent, error_recovery, role_playing, contingent, 
    - proactive, fluent, diverse, depth, likeable, 
    - understanding, flexible, informative, specific, relevant
    """
    querys = []
    refs = []
    hyps = []
    human_scores = [[] for _ in range(18)]  # One list for each attribute
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for dialogue in data:
            context = dialogue['context']
            response = dialogue['response']
            # FED doesn't have reference responses, so we'll use empty list
            reference = []
            
            # Get scores for each attribute
            scores = [
                dialogue['engaging'],
                dialogue['interesting'],
                dialogue['uses_knowledge'],
                dialogue['inquisitive'],
                dialogue['consistent'],
                dialogue['error_recovery'],
                dialogue['role_playing'],
                dialogue['contingent'],
                dialogue['proactive'],
                dialogue['fluent'],
                dialogue['diverse'],
                dialogue['depth'],
                dialogue['likeable'],
                dialogue['understanding'],
                dialogue['flexible'],
                dialogue['informative'],
                dialogue['specific'],
                dialogue['relevant']
            ]
            
            querys.append(context)
            refs.append([reference])  # Empty reference
            hyps.append(response)
            
            # Add scores for each attribute
            for i, score in enumerate(scores):
                human_scores[i].append(float(score))
    
    return querys, refs, [hyps], human_scores

# Update eval_metric.py to support FED
def eval_metric(args):
    # Add FED dataset handling
    if args.task_type == 'dialogue':
        if args.dataset == 'fed':
            querys, refs, hyps, human_scores = read_fed_data(args.data_path)
            # No need to average scores as they're already per attribute
        else:
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

    # Run FBD evaluation
    system_scores = []
    print("#-------------------------------------#")
    print(f"{args.metric} {args.model_type}")
    print("#-------------------------------------#")
    
    if args.metric == 'fbd':
        source_querys = querys
        source_answer_list = hyps
        target_querys, target_answers = prepare_data(querys, refs)
        tokenizer, model = get_modern_model_configs(args.model_type, args.embedding_model, args.is_chinese)

        mu1, sigma1 = get_statistics(target_querys, target_answers, tokenizer, 
                                   model, args.batch_size, use_cuda=True,
                                   embedding_model=args.embedding_model)
        
        for source_answers in source_answer_list:
            mu2, sigma2 = get_statistics(source_querys, source_answers, tokenizer, 
                                       model, args.batch_size, use_cuda=True,
                                       embedding_model=args.embedding_model)
            score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            system_scores.append(score)

    # Calculate correlations for each attribute
    print("\nCorrelations for each dialogue attribute:")
    attributes = [
        "Engaging", "Interesting", "Uses Knowledge", "Inquisitive",
        "Consistent", "Error Recovery", "Role Playing", "Contingent",
        "Proactive", "Fluent", "Diverse", "Depth", "Likeable",
        "Understanding", "Flexible", "Informative", "Specific", "Relevant"
    ]
    
    for attr_idx, attr_name in enumerate(attributes):
        pearson_corrs = []
        spearman_corrs = []
        scores = human_scores[attr_idx]
        pearson_corrs.append(abs(pearsonr(system_scores, scores)[0]))
        spearman_corrs.append(abs(spearmanr(system_scores, scores)[0]))
        print(f"\n{attr_name}:")
        print(f'Pearson correlation: {pearson_corrs}')
        print(f'Spearman correlation: {spearman_corrs}')

# Update argument parser
parser.add_argument('--dataset', type=str, default='convai2',
                   choices=['convai2', 'dailyh', 'dailyz', 'empathetic', 
                           'personam', 'personaz', 'fed'],
                   help='which dataset to evaluate on')
