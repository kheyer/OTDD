import sys
sys.path.append('../OTDD/')

import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from otdd import *
from otdd_pytorch import *
from mnist_helper import *
import os
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()

    if args.collect:

        distance_tensorized = PytorchEuclideanDistance()
        routine_tensorized = SinkhornTensorized(distance_tensorized)
        cost_tensorized = SamplesLossTensorized(routine_tensorized)

        # outputs = cost_tensorized.distance(mnist_sample.features, usps_sample.features, return_transport=True)
        # breakpoint()

        # reading all files in folder 'embeddings'
        file_path = '/home/ubuntu/efs/shared_folder/ModelSelection/data/embeddings'
        files = os.listdir(file_path)
        files.sort()

        # reading all embeddings
        embeddings = []
        file_names = []
        for idx, file in enumerate(files):
            # file_name equals to substring in file before "_text_results.csv"
            file_name = file.split('_text_results.csv')[0]
            if file_name == 'xcopa_et':
                continue
            file_names.append(file_name)
            with open(file_path + '/' + file, 'r') as f:
                s = f.read()
                s = s.split('\n')[:-1]
                s = [[float(x) for x in xs.split(',')] for xs in s]
                s = torch.tensor(s)
                embeddings.append(s)
                f.close()

        # compute the cost_tensorized.distance for all embeddings
        cost_tensorized_distances = []
        for i in range(len(embeddings)):
            distance_tmp = []
            for j in range(len(embeddings)):
                if i == j:
                    distance_tmp.append(0)
                else:
                    distance_tmp.append(cost_tensorized.distance(embeddings[i], embeddings[j])[0])
            cost_tensorized_distances.append(distance_tmp)
        # convert the cost_tensorized_distances to a numpy array
        cost_tensorized_distances = np.array(cost_tensorized_distances)

        # convert the cost_tensorized_distances to a pandas dataframe with both rows and columns being file_names
        # and the cost_tensorized_distances as values
        cost_tensorized_distances_df = pd.DataFrame(cost_tensorized_distances, index=file_names, columns=file_names)
        cost_tensorized_distances_df.to_csv('/home/ubuntu/efs/shared_folder/ModelSelection/data/otd_tensorized.csv')
        print('Distance Collected')
    
    
    model = SentenceTransformer('princeton-nlp/sup-simcse-roberta-large')
    # read in /home/ubuntu/efs/shared_folder/ModelSelection/data/otd_tensorized.csv
    distances = pd.read_csv('/home/ubuntu/efs/shared_folder/ModelSelection/data/otd_tensorized.csv')
    # read in /home/ubuntu/efs/shared_folder/ModelSelection/description_finetune.csv
    discriptions = pd.read_csv('/home/ubuntu/efs/shared_folder/ModelSelection/description_finetune.csv')
    # set dataset as the index of description
    descriptions = discriptions.set_index('dataset')
    distances = distances.set_index('Unnamed: 0')
    sets = descriptions.index.tolist()
    # retail rows in distances whose index is in sets
    distances = distances.loc[distances.index.isin(sets)]
    # retail columns in distances who are in the index of distance
    distances = distances.loc[:, distances.columns.isin(distances.index)]
    # retail rows in description whose dataset is in distance index
    descriptions = descriptions.loc[descriptions.index.isin(distances.index)]
    # concatenate the two dataframes by index
    distances = pd.concat([distances, descriptions], axis=1)

    for ds in distances.index.to_list():
        # get the row ds in distances
        ds_row = distances.loc[ds]
        distances[f'{ds}_residual'] =  abs(distances['zero'] - ds_row['zero'])
        # distances[f'{ds}_residual'] =  abs(distances['prefix'] - ds_row['prefix'])
    
    correlations = []
    names = []
    for ds in distances.index.to_list():
        # get the column ds in distances
        ds_distance = distances[ds]
        ds_residual = distances[f'{ds}_residual']
        ds_distance = ds_distance.drop(ds)
        ds_residual = ds_residual.drop(ds)
        # compute the correlation between ds_distance and ds_residual
        correlation = ds_distance.corr(ds_residual)
        correlations.append(correlation)
        names.append(ds)
    # print out a table, with the first row being names, and the second row being correlations
    print(pd.DataFrame([names, correlations]))

    data_descriptions = descriptions['description']
    text_embs = []
    for t in data_descriptions:
        # embed t using sentence_transformer
        text_embs.append(model.encode(t))
    distance_tmp = []
    for i in range(len(text_embs)):
        tmp = []
        for j in range(len(text_embs)):
            if i == j:
                tmp.append(0)
            else:
                tmp.append(cosine_similarity(text_embs[i].reshape(1, -1), text_embs[j].reshape(1, -1)).tolist()[0][0])
        distance_tmp.append(tmp)
    
    for idx, ds in enumerate(distances.index.to_list()):
        distances[f'{ds}_text'] = distance_tmp[idx]
    correlations = []
    names = []
    for ds in distances.index.to_list():
        # get the column ds in distances
        ds_distance = 1-distances[f'{ds}_text']
        ds_residual = distances[f'{ds}_residual']
        # delete ds row in ds_distance and ds_residual
        ds_distance = ds_distance.drop(ds)
        ds_residual = ds_residual.drop(ds)
        # compute the correlation between ds_distance and ds_residual
        correlation = ds_distance.corr(ds_residual)
        correlations.append(correlation)
        names.append(ds)
    # print out a table, with the first row being names, and the second row being correlations
    print(pd.DataFrame([names, correlations]))
    