#! -*- coding: utf-8 -*-

import os
import pickle
from scipy.sparse import csc_matrix
from scipy.linalg import block_diag
import dgl
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
#from HAN.newmodel import HAN
from BAKEHANmodel import HAN
from HAN.utils import load_data, EarlyStopping

import torch
from sklearn.metrics import f1_score
from DataShart import DataStriping
import numpy as np
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
#device = torch.device('cpu')
#torch.cuda.empty_cache()
def set_random_seed(seed=1):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_random_seed(1)

def Cat(feart):
    bert = BertModel.from_pretrained('../BERT').to(device)
    allids = []
    alltoken = []
    allatt = []
    for encoded_input in feart:

        allids.append(encoded_input['input_ids'])
        alltoken.append(encoded_input['token_type_ids'])
        allatt.append(encoded_input['attention_mask'])

        #tokenerlist.append(encoded_input)

    id = torch.cat(allids, dim=0)
    #id = torch.squeeze(id, dim=1)
    token = torch.cat(alltoken, dim=0)
    #token = torch.squeeze(token, dim=1)
    att = torch.cat(allatt, dim=0)
    #att = torch.squeeze(att, dim=1)
    # print(id.shape)
    # print(token.shape)
    # print(att.shape)
    input = {}
    input['input_ids'] = id.to(device)
    input['token_type_ids'] = token.to(device)
    input['attention_mask'] = att.to(device)

    bert.eval()
    with torch.no_grad():
        outputs = bert(**input)
    last_hidden_states = outputs.last_hidden_state
    sencefeat = last_hidden_states[:, 0, :]
    return sencefeat

#BERT model was used for feature extraction
def BERTTOKEN(feat):
    tokenizer = BertTokenizer.from_pretrained('../BERT')

    tokenerlist = []
    allids = []
    alltoken = []
    allatt = []
    for sence in feat:
        encoded_input = tokenizer.encode_plus(
            sence,
            padding='max_length',  
            max_length=20,  
            truncation=True,
            return_tensors='pt'
        )

        allids.append(encoded_input['input_ids'])
        alltoken.append(encoded_input['token_type_ids'])
        allatt.append(encoded_input['attention_mask'])

        tokenerlist.append(encoded_input)
        #alltokenfeat.append(sencefeat)

    id = torch.stack(allids, dim=0)
    id = torch.squeeze(id, dim=1)
    token = torch.stack(alltoken, dim=0)
    token = torch.squeeze(token, dim=1)
    att = torch.stack(allatt, dim=0)
    att = torch.squeeze(att, dim=1)
    # print(id.shape)
    # print(token.shape)
    # print(att.shape)
    input = {}
    input['input_ids'] = id.to(device)
    input['token_type_ids'] = token.to(device)
    input['attention_mask'] = att.to(device)

    return input
#selected_keys = ['1269', '1882', '1473', '1367']



#Construct heterogeneous graph structure
def hereroList(use2use,usefeatuse):
    allhetero = {}
    for key in use2use:
        csc1 = csc_matrix(use2use[key])
        csc2 = csc_matrix(usefeatuse[key])
        dict = {
            ('use', 'uu', 'use'): ('csc', (torch.tensor(csc1.indptr).long(), torch.tensor(csc1.indices).long(), torch.tensor(csc1.data).long())),
            ('use', 'uf', 'use'): ('csc', (torch.tensor(csc2.indptr).long(), torch.tensor(csc2.indices).long(), torch.tensor(csc2.data).long())),
        }

        hg = dgl.heterograph(dict, idtype=torch.int32)
        allhetero[key] = hg

    #print(allhetero)
    return  allhetero

#feature design
def Featlist(useoffeat):
    """

    :param useoffeat: User friend group User attributes
    :return: Usefriendfeat,allnumfeat
    """
    Usefriendfeat = {}
    allnumfeat={}

    for key in useoffeat:
        friendfeat = useoffeat[key]
        ftoken = BERTTOKEN(friendfeat)
        #friend = feat + list(friendfeat)
        #FBtoken = BERTTOKEN(friend)
        Usefriendfeat[key] = ftoken
        #allnumfeat[key] = FBtoken
        print(key)
    #print(Usefriendfeat)
    return Usefriendfeat

#Usefriendfeat,allnumfeat = Featlist(useoffeat)#
#allhetero = hereroList(use2use,usefeatuse)#

#print(allhetero)

def Datahecheng(use2use,usefeatuse,useaause,hyperedge,useoffeat,lable):
    """

    :param use2use: Metapath 1
    :param usefeatuse: Metapath 2
    .......
    :param useoffeat: User friend circle user personal information
    :param lable: attribute tag
    :return: generated data
    """

    Usefriendfeat = Featlist(useoffeat)  #
    #allhetero = hereroList(use2use, usefeatuse)  #
    Data = []
    for key in use2use:
        if key=='1153':
            print(key)
        data = []
        data.append(use2use[key])
        data.append(usefeatuse[key])
        data.append(useaause[key])
        data.append(hyperedge[key])
        data.append(Usefriendfeat[key])
        #data.append(allnumfeat[key])
        data.append(lable[key])
        Data.append(data)
    #print(Data)
    print("------完成数据加载————————")
    return Data




class MyGraphDataset(Dataset):
    def __init__(self, Data):
        self.data = Data

    def __len__(self):
        return len(self.data)
    def sym_norma(self,g):
        g1 = torch.from_numpy(g)
        diagonal_ones = torch.diag(torch.ones(g1.size(0)))
        g1 = g1 + diagonal_ones
        g1 = g1 + g1.t()
        g1[g1 > 1] = 1
        #g = g1[576:,576:]
        g = g1
        # Z1 = torch.diag(g1.sum(1))
        # D_inv_sqrt = torch.exp(-0.5 * torch.log(Z1 + 1e-5))
        # A_sym_normalized1 = torch.matmul(D_inv_sqrt, g1)
        # A_sym_normalized2 = torch.matmul(A_sym_normalized1, D_inv_sqrt)
        return g

    def __getitem__(self, idx):
        D = self.data[idx]
        g1 = D[0]
        g1A = self.sym_norma(g1)
        g2 = D[1]
        g2A = self.sym_norma(g2)
        g3 = D[2]
        g3A = self.sym_norma(g3)
        g4 = D[3]
        g4A = g4
        features = D[4]  
        labels =  D[5]
        return g1A,g2A,g3A,g4A,features,labels

# def BEACH(graph1,graph2):
#
#     #print(graph1)
#     #print(graph2)
#     new_graph1 = block_diag(*graph1)
#     new_graph2 = block_diag(*graph2)
#     csc1 = csc_matrix(new_graph1)
#     csc2 = csc_matrix(new_graph2)
#     dict = {
#         ('use', 'uu', 'use'): (
#         'csc', (torch.tensor(csc1.indptr).long(), torch.tensor(csc1.indices).long(), torch.tensor(csc1.data).long())),
#         ('use', 'uf', 'use'): (
#         'csc', (torch.tensor(csc2.indptr).long(), torch.tensor(csc2.indices).long(), torch.tensor(csc2.data).long())),
#     }
#
#     hg = dgl.heterograph(dict, idtype=torch.int32)
#
#     dantu = []
#     for i in range(len(graph1)):
#         csc3 = graph1[i]
#         csc4 = graph2[i]
#         csc3 = csc_matrix(csc3)
#         csc4 = csc_matrix(csc4)
#         dict = {
#             ('use', 'uu', 'use'): (
#                 'csc',
#                 (torch.tensor(csc3.indptr).long(), torch.tensor(csc3.indices).long(), torch.tensor(csc3.data).long())),
#             ('use', 'uf', 'use'): (
#                 'csc',
#                 (torch.tensor(csc4.indptr).long(), torch.tensor(csc4.indices).long(), torch.tensor(csc4.data).long())),
#         }
#         danhg = dgl.heterograph(dict, idtype=torch.int32)
#         dantu.append(danhg)
#
#
#
#
#     return hg,dantu

def BEACH(graph1,graph2,graph3,graph4):
    dantu = []
    for i in range(len(graph1)):
        csc3 = graph1[i]
        csc4 = graph2[i]
        csc5 = graph3[i]
        csc6 = graph4[i]

        csc3 = csc_matrix(csc3)
        csc4 = csc_matrix(csc4)
        csc5 = csc_matrix(csc5)
        csc6 = csc_matrix(csc6)
        dict = {
            ('use', 'uu', 'use'): (
                'csc',
                (torch.tensor(csc3.indptr).long(), torch.tensor(csc3.indices).long(), torch.tensor(csc3.data).long())),
            ('use', 'uf', 'use'): (
                'csc',
                 (torch.tensor(csc4.indptr).long(), torch.tensor(csc4.indices).long(), torch.tensor(csc4.data).long())),
            ('use', 'uaf', 'use'): (
                'csc',
                (torch.tensor(csc5.indptr).long(), torch.tensor(csc5.indices).long(), torch.tensor(csc5.data).long())),
            ('use', 'hyper', 'use'): (
                'csc',
                (torch.tensor(csc6.indptr).long(), torch.tensor(csc6.indices).long(), torch.tensor(csc6.data).long()))
            }
        danhg = dgl.heterograph(dict, idtype=torch.int32).to(device)
        dantu.append(danhg)
    BBhg = dgl.batch(dantu)
    return BBhg

# Defines the collate_fn function to group a batch of graphs together
def collate_fn(batch):
    graph1,graph2,graph3,graph4,faeture,lable = map(list, zip(*batch))
    #graph,faeture,lable = GuiYi(graphs, features, labels)
    #print(faeture.shape)

    Nodelist =[]
    for g in graph1:
        num = g.shape[0]
        Nodelist.append(num)


    batched_graph = BEACH(graph1,graph2,graph3,graph4)
    #print(batched_graph.number_of_nodes())
    batched_features = Cat(faeture)
    #print(batched_features.shape)
    lable1=[]
    for i in lable:
        combined_array = np.concatenate(i, axis=0)

        #encoded_labels = np.eye(2)[combined_array]
        torch_tensor = torch.from_numpy(combined_array)
        tensor_unsqueezed = torch_tensor.unsqueeze(0)
        lable1.append(tensor_unsqueezed)
    # for i in lable:
    #      torch_tensor = torch.from_numpy(i[0])
    #      tensor_unsqueezed = torch_tensor.unsqueeze(0)
    #      lable1.append(tensor_unsqueezed)

    batched_labels = torch.cat(lable1,dim=0)

    return batched_graph, batched_features, batched_labels,Nodelist

def score(logits, labels): #  micro_f1 和 macro_f1

    prediction = torch.gt(logits,0).float()
    labels =labels

    #print(prediction)
    #print(labels)

    fz =2 * torch.sum(labels * prediction).float()
    fm =torch.sum(labels + prediction).float()
    #print(fz)
    #print(fm)
    f1= fz / fm
    return f1, f1, f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e-12
    y_pred_pos = y_pred - (1 - y_true) * 1e-12  # 这里应该使用.clone()方法
    y_pred_pos = y_pred_pos.clone()  # 添加这行代码

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    loss = (neg_loss + pos_loss).mean()
    Loss = loss.requires_grad_(True)

    return Loss

if __name__  == '__main__':
    rate = 0.87
    selected_keys = ['1269','1882','1473','1367','1610','1377','1158','992','1503','1273']
    #selected_keys = ['1269', '1882']
    print(torch.cuda.is_available())
    #device =  torch.device('cuda0')
    #print(device)
    # Read the user's friends
    with open("../data/Facebook/u2u1.pkl", 'rb') as f1:
        use2use = pickle.load(f1)
        #use2use = {key: use2use[key] for key in selected_keys if key in use2use}

    # Read user characteristics User adjacency matrix
    with open("../data/Facebook/ufu1.pkl", 'rb') as f2:
        usefeatuse = pickle.load(f2)
        #usefeatuse = {key: usefeatuse[key] for key in selected_keys if key in usefeatuse}
    # Read the indirect user path
    with open("../data/Facebook/uaau1.pkl", 'rb') as f3:
        useaause = pickle.load(f3)
    
    with open("../data/Facebook/hyperedge.pkl", 'rb') as f4:
        hyperedge = pickle.load(f4)

    with open("u2feat.pkl", 'rb') as f5:
        useoffeat = pickle.load(f5)
        #useoffeat = {key: useoffeat[key] for key in selected_keys if key in useoffeat}

    # 读取用户标签
    with open('lable.pkl', 'rb') as f6:
        lable = pickle.load(f6)
        #lable = {key: lable[key] for key in selected_keys if key in lable}
  
    # with open("feat.pkl", "rb") as f7:
    #     feat = pickle.load(f7)
    #data = Datahecheng(use2use, usefeatuse, useoffeat, lable)
  
    train,test = DataStriping(use2use,usefeatuse,useaause,hyperedge,useoffeat,lable,rate)
    trainData = Datahecheng(train[0],train[1],train[2],train[3],train[4],train[5])
    testData = Datahecheng(test[0], test[1], test[2], test[3], test[4], test[5])
    traindataset = MyGraphDataset(trainData)
    testdataset = MyGraphDataset(testData)
    batch_size = 32
    traindataloader = DataLoader(traindataset, batch_size=batch_size, collate_fn=collate_fn)
    testdataloader = DataLoader(testdataset, batch_size=batch_size, collate_fn=collate_fn)

    model = HAN(meta_paths=[["uu"], ["uf"],["uaf"],["hyper"]],
                 in_size=768,
                 hidden_size=384,
                 out_size=768,
                 num_heads= [4],
                 dropout=0.5).to(device)


    loss_fcn = torch.nn.CrossEntropyLoss()
    loss_BCfcn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    for epch in range(460):
        model.train()
        allacc =[]
        f1mi =[]
        f1ma =[]
        allloss =[]
        alllogits = []
        j=0
        for step, (batched_graph, batched_features, batched_labels,Nodelist) in enumerate(traindataloader):
            j=j+1
            batched_graph1 = batched_graph.to(device)
            batched_labels1 =batched_labels.type(torch.float32).to(device)
            batched_features1 = batched_features.to(device)
            optimizer.zero_grad()
            logits = model(batched_graph1,batched_features1)
            loss = multilabel_categorical_crossentropy(batched_labels1,logits)
            allloss.append(loss)
            accuracy, micro_f1, macro_f1=score(logits, batched_labels1)
            allacc.append(accuracy)
            f1ma.append(macro_f1)
            f1mi.append(micro_f1)
            loss.backward()
            optimizer.step()

        print(step)
        loss1 = sum(allloss)/j
        acc = sum(allacc)/j
        f1ma = sum(f1ma)/j
        f1mi =sum(f1mi)/j
        print("---------train-loss-------",loss1.item(),"-----train-F1-----",f1ma.item(),"-----次数-----",epch)
        model.eval()
        with torch.no_grad():
            allacc1 = []
            f1mi1 = []
            f1ma1 = []
            allloss1 = []
            alllogits1 = []
            j = 0
            for step, (batched_graph, batched_features, batched_labels, Nodelist) in enumerate(testdataloader):
                j = j + 1
                batched_graph1 = batched_graph.to(device)
                batched_labels1 = batched_labels.type(torch.float32).to(device)
                batched_features1 = batched_features.to(device)
                logits = model(batched_graph1, batched_features1)
                loss = multilabel_categorical_crossentropy(batched_labels1, logits)
                allloss1.append(loss)
                accuracy, micro_f1, macro_f1 = score(logits, batched_labels1)
                allacc1.append(accuracy)
                f1ma1.append(macro_f1)
                f1mi1.append(micro_f1)

            print(step)
            loss1 = sum(allloss1) / j
            acc = sum(allacc1) / j
            f1ma = sum(f1ma1) / j
            f1mi = sum(f1mi1) / j
            print("---------test-loss-------", loss1.item(), "-----test-F1-----", f1ma.item(), "-----次数-----", epch)

