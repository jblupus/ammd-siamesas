import tensorflow as tf; print(tf.__version__)
import numpy as np
import os
import random

def load_img(image_file):
    return tf.io.read_file(image_file)

def img_to_array(image_raw):
    return tf.image.decode_image(image_raw).numpy()

def list_cam_files(cam_files_path):
    return [filename for filename in os.listdir(cam_files_path) if '.' not in filename]

def list_person_files(person_files_path):
    return [filename for filename in os.listdir(person_files_path) if '.png' in filename and '_' not in filename]

def map_person_files(data_path):
    cam = list_cam_files(data_path)
    return {person: [data_path + person + '/'] for person in cam}

def data_generator(data_folder, dataset_size=1000):
    data_path_cam0 = data_folder + '0/'
    data_path_cam1 = data_folder + '1/'
    cam = map_person_files(data_path_cam0)
    cam1 = map_person_files(data_path_cam1)
    
    persons = [person for person in cam.keys()]
    
    for person in persons:
        cam[person].extend(cam1[person])
        
    def get_negative(person, persons):
        while True:
            _person = random.choice(persons)
            if _person != person:
                return _person
    
    def choose_file(data_path, ref_file=None):
        while True:
            img = data_path + random.choice(list_person_files(data_path))
            if img != ref_file:
                return img
        
    cams = [0, 1]
    X_data = []
    Y_data = []
    for _ in range(dataset_size):
        person = random.choice(persons)
        nperson = get_negative(person, persons)
        
        anchor_person = random.choice(cams)
        pos_person = random.choice(cams)
        neg_person = random.choice(cams)
        
        anchor_img = choose_file(cam[person][anchor_person])
        pos_img = choose_file(cam[person][pos_person], anchor_img)
        neg_img = choose_file(cam[nperson][neg_person])
        
        anchor_img = load_img(cam[person][anchor_person])
        pos_img = load_img(cam[person][pos_person], anchor_img)
        neg_img = load_img(cam[nperson][neg_person])
        
        anchor_img = img_to_array(cam[person][anchor_person])
        pos_img = img_to_array(cam[person][pos_person], anchor_img)
        neg_img = img_to_array(cam[nperson][neg_person])
        
        X_data.append([anchor_img, pos_img, neg_img])
        Y_data.append([person, nperson])
    return np.array(X_data), np.array(Y_data)


# train_data_path = './datasets/airport-alunos/treino/'
# X_data, y_data = data_generator(train_data_path, dataset_size=2)
# X_data
