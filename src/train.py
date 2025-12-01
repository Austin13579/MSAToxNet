from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef,roc_auc_score, average_precision_score,precision_score, f1_score
import pandas as pd
import argparse
import torch
import numpy as np
import random
import torch.nn as nn
from sklearn.utils import class_weight
import copy

from model import MSAToxNet
from utils import Data_Encoder



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# training function at each epoch
def train(model, loader, optimizer):
    print('Training on {} samples...'.format(len(loader.dataset)))
    model.train()
    losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    for batch_idx, (fp, seq1,seq2,label) in enumerate(loader):
        optimizer.zero_grad()

        output = model(fp, seq1,seq2)
        score=torch.squeeze(output)
        loss = loss_fn(score, label.float())

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

        total_preds = torch.cat((total_preds, score.cpu()), 0)
        total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return np.mean(losses),total_labels.numpy().flatten(), total_preds.detach().numpy().flatten()


def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (fp, seq1,seq2, label) in enumerate(loader):
            output= model(fp, seq1,seq2)

            score = torch.squeeze(output)
            total_preds = torch.cat((total_preds, score.cpu()), 0)
            total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='Rabbit', help='which dataset')
    parser.add_argument('--rs', type=int, default=0, help='which random seed')
    args = parser.parse_args()
    setup_seed(42)
    print(args.ds)

    # Read datasets
    train_path='../dataset/datas/'+args.ds+'_train'+str(args.rs)+'.csv'
    train_df = pd.read_csv(train_path)

    valid_path='../dataset/datas/'+args.ds+'_valid'+str(args.rs)+'.csv'
    valid_df = pd.read_csv(valid_path)

    external_path='../dataset/'+args.ds+'_external.csv'
    external_df = pd.read_csv(external_path)

    tox_model=MSAToxNet()

    batch_size = 64
    LR = 1e-5
    NUM_EPOCHS = 50


    train_set = Data_Encoder(train_df.index.values, train_df)
    valid_set = Data_Encoder(valid_df.index.values, valid_df)
    external_set = Data_Encoder(external_df.index.values, external_df)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    external_loader = torch.utils.data.DataLoader(external_set, batch_size=batch_size, shuffle=False)

    train_label=train_df.Label
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_label),y=train_label)
    class_weights_dict = dict(enumerate(class_weights))

    p_weight=class_weights_dict[1]/class_weights_dict[0]
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p_weight]))
    optim = torch.optim.Adam(tox_model.parameters(), lr=LR)

    best_auc=0
    res={}
    val,tmp=[],[]
    for epoch in range(NUM_EPOCHS):
        print("Epoch: ", epoch + 1)
        train_loss,aa,bb = train(tox_model, train_loader, optim)
        print("Train Loss: ", train_loss)

        print("Validation")
        valid_true, valid_prob = predicting(tox_model, valid_loader)
        v_auroc = roc_auc_score(valid_true, valid_prob)
        v_auprc=average_precision_score(valid_true, valid_prob)
        print("Val AUPRC: ", v_auprc)

        test_true, test_prob = predicting(tox_model, external_loader)
        test_pred = (test_prob >= 0.5).astype(int)


        tn, fp, fn, tp = confusion_matrix(external_df['Label'], test_pred).ravel()
        specificity = tn / (tn + fp)

        res = {
            'Accuracy': accuracy_score(external_df['Label'], test_pred),
            'AUROC': roc_auc_score(external_df['Label'], test_prob),
            'AUPRC': average_precision_score(external_df['Label'], test_prob),
            'Precision': precision_score(external_df['Label'], test_pred),
            'Recall': recall_score(external_df['Label'], test_pred),
            'F1': f1_score(external_df['Label'], test_pred),
            'MCC': matthews_corrcoef(external_df['Label'], test_pred),
            'Specificity': specificity
        }
        val.append([v_auprc,v_auroc])
        tmp.append((res, copy.deepcopy(tox_model.state_dict())))
        print(res)

    ## Model average
    # Best and second-best models based on AUPRC
    jj,kk=torch.topk(torch.tensor(val)[:,0],k=2)
    best_queue=[]
    for k in kk:
        best_queue.append(tmp[k])

    # Best model based on AUROC
    mm,nn=torch.topk(torch.tensor(val)[:,1],k=1)
    best_queue.append(tmp[nn[0]])
    
    # Load the average weights to a new model
    new_model=MSAToxNet()
    weighted_params = {}
    for key in best_queue[0][1].keys():
        weighted_params[key] = torch.zeros_like(best_queue[0][1][key])+1e-10

    weight=[0.8,0.1,0.1]
    for i,(res,state_dict) in enumerate(best_queue):
        print(res)
        for key in state_dict:
            weighted_params[key] += weight[i]*state_dict[key]

    for key in best_queue[0][1].keys():
        weighted_params[key] -= 1e-10


    new_model.load_state_dict(weighted_params)
    test_true, test_prob = predicting(new_model, external_loader)
    test_pred = (test_prob >= 0.5).astype(int)


    results = {
        'Accuracy': accuracy_score(external_df['Label'], test_pred),
        'AUROC': roc_auc_score(external_df['Label'], test_prob),
        'AUPRC': average_precision_score(external_df['Label'], test_prob),
        'Recall': recall_score(external_df['Label'], test_pred),
        'F1': f1_score(external_df['Label'], test_pred),
        'MCC': matthews_corrcoef(external_df['Label'], test_pred)
    }
    print()
    print(results)
    pd.DataFrame([results]).to_csv('results/results_' + args.ds + str(args.rs) + '.csv',
                                    columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)
