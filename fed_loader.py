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
