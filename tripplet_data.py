import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt


def filter_possible(pid1, pids2, same=True):
    person_number = pid1[0]
    if same is True:
        possible_ids2 = [pid2 for pid2 in pids2 if pid2[0] == person_number]
    else:
        possible_ids2 = [pid2 for pid2 in pids2 if pid2[0] != person_number]
    return possible_ids2

def combine_cam_files(pids1, pids2):
    i = 0
    data = {}
    def negative(i, pid1): 
        if i % 2 == 0:
            return (0, *random.choice(filter_possible(pid1, pids1, same=False)))
        else:
            return (1,  *random.choice(filter_possible(pid1, pids2, same=False)))
        
    for i, pid1 in enumerate(pids1):
        data[(0, *pid1)] = { 
            'pos': (1, *random.choice(filter_possible(pid1, pids2, same=True))),
            'neg': negative(i, pid1) }
    return data

def pairs_labels(d_combination, batch):
    same = lambda k1, k2: 1. if k1==k2 else 0.
    for _ in range(batch):
        for key, values in d_combination.items():
            neg_value = random.choice(values['neg'])
            yield(key, values['pos'], values['neg'])
            
def generate_data(cam_img_dict, pids1, pids2, batch=1):
    
    X_anchor = [] 
    X_positive = []
    X_negative = []
    
    for _ in range(batch):
        combination_data = combine_cam_files(pids1, pids2)
        data = list(pairs_labels(combination_data, batch))
        X_anchor.extend([cam_img_dict[x[0]] for x in data])
        X_positive.extend([cam_img_dict[x[1]] for x in data])
        X_negative.extend([cam_img_dict[x[2]] for x in data])
        
    X_anchor = np.array(X_anchor, dtype = np.float32)
    X_positive = np.array(X_positive, dtype = np.float32)
    X_negative = np.array(X_negative, dtype = np.float32)
    
    return X_anchor, X_positive, X_negative 