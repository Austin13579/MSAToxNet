import numpy as np
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import RDLogger
import torch
from tokenizers import Tokenizer


RDLogger.DisableLog('rdApp.*')


# Drug dictionary
drug_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


tokenizer = Tokenizer.from_file("tokenizer.json")


def encode_drug(drug_seq, drug_dic):
    max_drug =80
    e_drug = [drug_dic[aa] for aa in drug_seq]
    ld = len(e_drug)
    if ld < max_drug:
        v_d = np.pad(e_drug, (0, max_drug - ld), 'constant', constant_values=0)
    else:
        v_d = e_drug[:max_drug]
    return v_d


def encode_drug2(drug_seq):
    max_drug = 40

    encoding = tokenizer.encode_batch([drug_seq])
    n_drug = encoding[0].ids
    e_drug=[]
    for n in n_drug:
        if n in [1, 2, 3]:
            continue
        elif n in [0]:
            e_drug.append(497)
        else:
            e_drug.append(n - 3)
    ld = len(e_drug)
    if ld < max_drug:
        v_d = np.pad(e_drug, (0, max_drug - ld), 'constant', constant_values=0)
    else:
        v_d = e_drug[:max_drug]
    return v_d



class Data_Encoder(torch.utils.data.Dataset):

    def __init__(self, data_id, all_data):
        """Initialization."""
        self.all_data = all_data
        self.data_id = data_id


    def __len__(self):
        """Get size of input data."""
        return len(self.data_id)

    def __getitem__(self, index):
        """Get items from raw data."""
        index = self.data_id[index]

        smile,label=self.all_data.iloc[index].iloc[1],self.all_data.iloc[index].iloc[0]
        mol = Chem.MolFromSmiles(smile)


        fp1 = MACCSkeys.GenMACCSKeys(mol).ToList()
        fp2=AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048).ToList()
        fp = np.concatenate((fp1, fp2))
        

        atom_seq=encode_drug(smile,drug_dict)
        sub_seq=encode_drug2(smile)

        return np.asarray(fp).astype('float32'), np.asarray(atom_seq), np.asarray(sub_seq),label
