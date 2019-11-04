import tensorflow as tf

from tqdm import tqdm
import random

import numpy as np

import matplotlib.pyplot as plt

def load_img(image_file):
    return tf.io.read_file(image_file)

def img_to_array(image_raw):
    return tf.image.decode_image(image_raw).numpy()

def person_to_img(file_names):
    cam_path_to_img = {}
    pids = [[], []]
    img_counts = [0, 0]
    for f in list(file_names):
        cid = int(f.split('/')[-3])
        pid = int(f.split('/')[-2])
        iid = int(f.split('/')[-1][:3])
        pids[cid].append((pid, iid))
        cam_path_to_img[cid, pid, iid] = img_to_array(load_img(f))
        img_counts[cid] += 1
    print('Images in cam1 = %d, cam2 = %d'%(img_counts[0], img_counts[1]))
    # dic person_to_img e person ids em cada camera
    return cam_path_to_img, list(set(pids[0])), list(set(pids[1]))

def filter_possible(pid1, pids2, same=True):
    person_number = pid1[0]
    if same is True:
        possible_ids2 = [pid2 for pid2 in pids2 if pid2[0] == person_number]
    else:
        possible_ids2 = [pid2 for pid2 in pids2 if pid2[0] != person_number]
    return possible_ids2

def combine_cam_files(pids1, pids2):
    i = 0
    d = {}
    for pid1 in tqdm(pids1):
        if i % 2 == 0:
            d[pid1] = filter_possible(pid1, pids2, same=True)
        else:
            d[pid1] = filter_possible(pid1, pids2, same=False)
        i += 1
    return d

def get_batch(d_combination, cam_img_dict):
    def pairs_labels():
        i = 0        
        for key, value in d_combination.items():
            if i % 2 == 0:
                t = ((0, *key), (1, *random.choice(value)), 1.)
            else:
                t = ((0, *key), (1, *random.choice(value)), 0.)
            i += 1
            yield t
            
    train_data = list(pairs_labels())
    
    X_a_train = np.array([cam_img_dict[x[0]] for x in train_data], dtype = np.float32)
    X_b_train = np.array([cam_img_dict[x[1]] for x in train_data], dtype = np.float32)
    
    y_train = np.array([x[2] for x in train_data], dtype = np.float32)
    
    return X_a_train, X_b_train, y_train

# plot pics side by side
def show_side_by_side(figs, titles = None, limit = 10, figsize=(20, 4), cmap = 'gray', grid = False):
    minval = min(limit, figs.shape[0])
    plt.figure(figsize = figsize)
    for i in range(minval):
        subplot = plt.subplot(1, limit, i + 1)
        extent = (0, figs[i].shape[1], figs[i].shape[0], 0)
        subplot.imshow(figs[i], cmap = cmap, extent = extent)
        if titles:
            subplot.set_title(titles[i])
        if grid:
            subplot.grid(color='gray', linestyle='-', linewidth=1)
        else:
            subplot.get_xaxis().set_visible(False)
            subplot.get_yaxis().set_visible(False)
    plt.show()
    
def plot_sample(cam1, cam2, val):
    rindices = np.random.choice(np.array(range(cam1.shape[0])), 10, replace=False)
    show_side_by_side(cam1[rindices], titles = [('DIF' if n == 0 else 'IGUAL') for n in val[rindices]])
    show_side_by_side(cam2[rindices])