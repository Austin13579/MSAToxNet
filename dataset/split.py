import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='Rabbit', help='which dataset')
parser.add_argument('--rs', type=int, default=0, help='which random seed')
args = parser.parse_args()

dataset = args.ds
seed = args.rs
np.random.seed(seed)
print("Dataset: " + dataset + ", random seed: " + str(seed))
df = pd.read_csv(dataset + '.csv')
df_pos=df[df['Label']==1]
df_neg=df[df['Label']==0]
df_pos = shuffle(df_pos)
df_neg = shuffle(df_neg)


if dataset=='Rabbit':
    train_p_num,train_n_num = 514,873
    train_p_data,valid_p_data = df_pos.iloc[:train_p_num],df_pos.iloc[train_p_num:]
    train_n_data,valid_n_data = df_neg.iloc[:train_n_num],df_neg.iloc[train_n_num:]

    df_train = pd.concat([train_p_data, train_n_data], axis=0)
    df_valid = pd.concat([valid_p_data, valid_n_data], axis=0)

    df_train.to_csv('datas/'+dataset + '_train' + str(seed) + '.csv', index=False)
    df_valid.to_csv('datas/'+dataset + '_valid' + str(seed) + '.csv', index=False)

elif dataset=='Rat':
    train_p_num,train_n_num = 451,892
    train_p_data,valid_p_data = df_pos.iloc[:train_p_num],df_pos.iloc[train_p_num:]
    train_n_data,valid_n_data = df_neg.iloc[:train_n_num],df_neg.iloc[train_n_num:]

    df_train = pd.concat([train_p_data, train_n_data], axis=0)
    df_valid = pd.concat([valid_p_data, valid_n_data], axis=0)

    df_train.to_csv('datas/'+dataset + '_train' + str(seed) + '.csv', index=False)
    df_valid.to_csv('datas/'+dataset + '_valid' + str(seed) + '.csv', index=False)

else:
    raise ValueError("Unexpected Datasets")