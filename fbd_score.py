# Frechet Bert Distance (FBD)
import copy

from utils import (
  read_data,
  get_model_configs,
  get_embeddings,
  calculate_feature_statistics, 
  calculate_frechet_distance,
  transform_qa_pairs,
  read_dialogue
)

from eval_metric import (
  get_modern_embeddings,
  get_modern_model_configs
)

def get_statistics(
    querys, 
    answers, 
    tokenizer, 
    model, 
    batch_size, 
    use_cuda=True,
    embedding_model='roberta'  # Add default parameter
):
    feats = get_modern_embeddings(
        querys, 
        answers, 
        tokenizer, 
        model, 
        batch_size, 
        use_cuda,
        embedding_model
    )
    return calculate_feature_statistics(feats)

def calculate_fbd(
    source_querys,
    source_answers,
    target_querys,
    target_answers,
    is_chinese,
    pretrained_model_path,
    batch_size,
    device,
    embedding_model='roberta'  # Add default parameter
):
    tokenizer, model = get_modern_model_configs(pretrained_model_path, embedding_model, is_chinese)
    print('get statistics from source data ...')
    mu1, sigma1 = get_statistics(source_querys, source_answers, tokenizer, model, 
                                batch_size, use_cuda=(device=='gpu'),
                                embedding_model=embedding_model)
    print('get statistics from target data ...')
    mu2, sigma2 = get_statistics(target_querys, target_answers, tokenizer, model, 
                                batch_size, use_cuda=(device=='gpu'),
                                embedding_model=embedding_model)
    print('calculate FBD score ...')
    score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return score

def fbd_score(args):
    if args.source_path is not None and args.target_path is not None:
        source_querys, source_answers = read_data(args.source_path)
        target_querys, target_answers = read_data(args.target_path)
    elif args.data_path is not None:
        source_querys, source_answers, _, _ = read_dialogue(args.data_path)
        target_querys = copy.deepcopy(source_querys)
        target_answers = copy.deepcopy(source_answers)
    
    print(len(source_querys), len(source_answers))
    
    if args.transform:
        target_querys, target_answers = transform_qa_pairs(
            target_querys, 
            target_answers, 
            args.transform, 
            args.ratio, 
            args.noise_dict, 
            args.repeat_dict
        )
    
    return calculate_fbd(
        source_querys, 
        source_answers,
        target_querys,
        target_answers, 
        is_chinese=args.is_chinese,
        pretrained_model_path=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
        embedding_model=args.embedding_model  # Pass through the embedding model type
    )

if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_path', type=str, help='path to the file of question answer pair')
    parser.add_argument('--target_path', type=str, help='path to the file of question answer pair')
    parser.add_argument('--data_path', type=str, help='path to dialogue annotation data')
    parser.add_argument('--model_type', type=str, default='', help='pretrained model type or path to pretrained model')
    parser.add_argument('--is_chinese', type=int, default=0, help='Is Chinese corpus or not')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_model', type=str, default='roberta', 
                       choices=['roberta', 'deberta', 'bart', 't5'], 
                       help='Type of embedding model to use')
    parser.add_argument('--transform', type=str, default=None, 
                       help='transformation type for target pairs: [noise | mismatch | permutate | repeat]')
    parser.add_argument('--ratio', type=float, default=0.5, help='ratio of transformed pairs')
    parser.add_argument('--noise_dict', type=str, default=None, help='path to the noise dictionary')
    parser.add_argument('--repeat_dict', type=str, default=None, help='path to the repeatition dictionary')
    parser.add_argument('--device', type=str, default='cpu', help='[cpu | gpu]')
    
    args = parser.parse_args()
    fbd_score(args)
