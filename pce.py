import numpy as np
from numpy import pi
from scipy.special import digamma, loggamma
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import torch
from utils import (
    read_data,
    get_model_configs,
    get_embeddings,
    transform_qa_pairs,
    read_dialogue
)
import copy

def volume_of_unit_ball_log(d):
    """Calculate the log volume of a unit ball in d dimensions."""
    d_over_2 = d / 2
    return d_over_2 * np.log(pi) - (loggamma(d_over_2 + 1))

def cross_entropy(N, M, k, nu_k, d):
    """Calculate cross entropy estimate."""
    psi_k = digamma(k)
    c_bar = volume_of_unit_ball_log(d)
    inner_term = np.log(M) - psi_k + c_bar + d * np.log(nu_k)
    return (1 / N) * np.sum(inner_term)

def entropy(N, k, rho_k, d):
    """Calculate entropy estimate."""
    psi_k = digamma(k)
    c_bar = volume_of_unit_ball_log(d)
    inner_term = np.log(N-1) - psi_k + c_bar + d * np.log(rho_k)
    return (1 / N) * np.sum(inner_term)

def calculate_information_scores(source_embeddings, target_embeddings, k=2, C=3):
    """
    Calculate information theoretic scores between source and target embeddings.
    
    Args:
        source_embeddings: numpy array of source text embeddings
        target_embeddings: numpy array of target text embeddings
        k: number of nearest neighbors (default: 2)
        C: multiplier for precision calculation (default: 3)
    
    Returns:
        Dictionary containing various distance metrics
    """
    prc_k = C * k
    
    # Set up nearest neighbor graphs
    nbrs_source = NearestNeighbors(n_neighbors=prc_k+1, algorithm='auto', n_jobs=-1).fit(source_embeddings)
    dist_source, _ = nbrs_source.kneighbors(source_embeddings, k+1)
    
    nbrs_target = NearestNeighbors(n_neighbors=prc_k+1, algorithm='auto', n_jobs=-1).fit(target_embeddings)
    dist_target, _ = nbrs_target.kneighbors(target_embeddings, k+1)
    
    # Calculate cross-distances
    dist_source_target, _ = nbrs_target.kneighbors(source_embeddings, k+1)
    dist_target_source, _ = nbrs_source.kneighbors(target_embeddings, k+1)
    
    # Calculate information theoretic scores
    ce_ts = cross_entropy(
        len(target_embeddings), 
        len(source_embeddings), 
        k, 
        dist_target_source[:, k-1], 
        len(source_embeddings[0])
    )
    
    ce_st = cross_entropy(
        len(source_embeddings), 
        len(target_embeddings), 
        k, 
        dist_source_target[:, k-1], 
        len(source_embeddings[0])
    )
    
    e_s = entropy(
        len(source_embeddings), 
        k, 
        dist_source[:, k], 
        len(source_embeddings[0])
    )
    
    e_t = entropy(
        len(target_embeddings), 
        k, 
        dist_target[:, k], 
        len(target_embeddings[0])
    )
    
    return {
        'cross_entropy_ts': ce_ts - e_s,  # Target-source relative to source
        'cross_entropy_st': ce_st - e_s,  # Source-target relative to source
        'entropy_diff': e_t - e_s         # Entropy difference
    }

def calculate_itm(
    source_querys,
    source_answers,
    target_querys,
    target_answers,
    is_chinese,
    pretrained_model_path,
    batch_size,
    device,
    k=2,
    C=3
):
    """
    Calculate Information Theoretic Metric scores between source and target text pairs.
    """
    # Get model and tokenizer
    tokenizer, model = get_model_configs(pretrained_model_path, is_chinese)
    
    print('Getting embeddings from source data...')
    source_embeddings = get_embeddings(
        source_querys, 
        source_answers, 
        tokenizer, 
        model, 
        batch_size, 
        use_cuda=(device=='gpu')
    )
    
    print('Getting embeddings from target data...')
    target_embeddings = get_embeddings(
        target_querys, 
        target_answers, 
        tokenizer, 
        model, 
        batch_size, 
        use_cuda=(device=='gpu')
    )
    
    print('Calculating Information Theoretic scores...')
    scores = calculate_information_scores(
        source_embeddings.cpu().numpy(), 
        target_embeddings.cpu().numpy(),
        k=k,
        C=C
    )
    
    return scores

def itm_score(args):
    """Main function to calculate Information Theoretic Metric scores."""
    if args.source_path is not None and args.target_path is not None:
        source_querys, source_answers = read_data(args.source_path)
        target_querys, target_answers = read_data(args.target_path)
    elif args.data_path is not None:
        source_querys, source_answers, _, _ = read_dialogue(args.data_path)
        target_querys = copy.deepcopy(source_querys)
        target_answers = copy.deepcopy(source_answers)
    
    print(f"Processing {len(source_querys)} source and {len(target_querys)} target pairs")
    
    if args.transform:
        target_querys, target_answers = transform_qa_pairs(
            target_querys,
            target_answers,
            args.transform,
            args.ratio,
            args.noise_dict,
            args.repeat_dict
        )
    
    return calculate_itm(
        source_querys,
        source_answers,
        target_querys,
        target_answers,
        is_chinese=args.is_chinese,
        pretrained_model_path=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
        k=args.neighbors,
        C=args.precision_multiplier
    )

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_path', type=str, help='path to source question-answer pairs')
    parser.add_argument('--target_path', type=str, help='path to target question-answer pairs')
    parser.add_argument('--data_path', type=str, help='path to dialogue annotation data')
    parser.add_argument('--model_type', type=str, default='', help='pretrained model type or path')
    parser.add_argument('--is_chinese', type=int, default=0, help='is Chinese corpus (0/1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--neighbors', type=int, default=2, help='number of nearest neighbors (k)')
    parser.add_argument('--precision_multiplier', type=int, default=3, help='precision calculation multiplier (C)')
    parser.add_argument('--transform', type=str, default=None, 
                       help='transformation type: [noise|mismatch|permutate|repeat]')
    parser.add_argument('--ratio', type=float, default=0.5, help='transformation ratio')
    parser.add_argument('--noise_dict', type=str, default=None, help='path to noise dictionary')
    parser.add_argument('--repeat_dict', type=str, default=None, help='path to repeat dictionary')
    parser.add_argument('--device', type=str, default='cpu', help='device to use [cpu|gpu]')
    
    args = parser.parse_args()
    scores = itm_score(args)
    print("Information Theoretic Metric Scores:")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
