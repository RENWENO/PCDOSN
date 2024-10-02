#! -*- coding: utf-8 -*-


import pickle
import re
import numpy as np
from transformers import BertModel, BertTokenizer

def re1(String):
    ori =  String
    cleaned_string1 = re.sub(";anonymized", '', ori)
    cleaned_string2 = re.sub(";id", '', cleaned_string1)
    cleaned_string = re.sub(";", ' ', cleaned_string2)
    return  cleaned_string

def readfile(featnames,feat,u2u1,ufu1,edges):


    with open(featnames,"r",encoding="utf-8") as f4:
        featdict = {}
        attributedict = {}
        allfeatname = f4.readlines()
        Attributed = set()
        for allfeat in allfeatname:
            feat = allfeat.split(" ")
            num = int(feat[0])
            Avelue = re1(feat[1])
            Attributed.add(Avelue)
            attributedict[num] = Avelue
            Fvelue = feat[-2]+" "+feat[-1][:-1]
            featdict[num] = Fvelue
        Atttiedict={}
        for i,velu in enumerate(Attributed):
            Atttiedict[i]=velu
    #
    #
        print(attributedict)
        print(featdict)
        print(Attributed)


    with open(feat,'r',encoding="utf-8") as f5:
        allusefeat = f5.readlines()
        usefeatdict = {}
        for usefeat in allusefeat:
            AttriV = np.zeros((1,23))
            uf = usefeat[:-1].split(" ")
            key = uf[0]
            veul1 = uf[1:]
            veul = np.array([int(char) for char in uf[1:]])
            for j,V in enumerate(velu):
                if V == 1:
                    S=attributedict[j]
                    for k,v in Atttiedict:
                        if v == S:
                            key=k
            #attrie = np.
            usefeatdict[key]=veul
        print(usefeatdict)


    #u-u图
    with open(u2u1, 'rb') as f1:
        arrays_loaded1 = pickle.load(f1)
        keylist1 = []
        for key1 in arrays_loaded1.keys():
            keylist1.append(key1)


    #u-A|B-u
    with open(ufu1, 'rb') as f2:
        arrays_loaded2 = pickle.load(f2)
        keylist2 = []
        for key2 in arrays_loaded2.keys():
            keylist2.append(key2)
    # def usetofeat(usefrident):
    #     for

    with open(edges,"r") as f4:
        edgs = f4.readlines()
        friendlist = []
        friendstrlist =[]
        alllablelist=[]
        for u in keylist1:
            lablelist=[]
            uF = usefeatdict[u]
            lable1 = uF[264:266]#2
            lablelist.append(lable1)
            lable2 = uF[494:517]#23
            lablelist.append(lable2)
            lable3 = uF[220:223]#3
            lablelist.append(lable3)
            lable4 = uF[0:17]#16
            lablelist.append(lable4)
            usefrident = set()
            usefridentStr = []
            for edd in edgs:

                if u== edd[:-1].split(" ")[0]:
                    usefrident.add(edd[:-1].split(" ")[1])
                    F = edd[:-1].split(" ")[1]
                    Farr = usefeatdict[F]
                    indix = np.argwhere(Farr == 1)
                    indix = np.squeeze(indix, axis=1)
                    frientfeatstr = []
                    for id in indix:
                        featfr = featdict[id]
                        frientfeatstr.append(featfr)
                    print(frientfeatstr)
                    frientstr = " ".join(frientfeatstr)
                    usefridentStr.append(frientstr)



            friendlist.append(usefrident)
            friendstrlist.append(usefridentStr)
            alllablelist.append(lablelist)
    return  keylist1,friendstrlist,alllablelist,arrays_loaded1





def label_feat(keylist1,friendstrlist,alllablelist,arrays_loaded1):
    usefriefeat={}
    selfuselabe = {}
    for i,key in  enumerate(keylist1):
        velu = friendstrlist[i]
        velu1 = alllablelist[i]
        usefriefeat[key]=velu
        selfuselabe[key] = velu1

        if len(friendstrlist[i]) != arrays_loaded1[key].shape[0]-576:
            print(key, "Number of friends：", len(friendstrlist[i]))
            print(key, "Number of friends：", arrays_loaded1[key].shape[0] - 576)
    return selfuselabe,usefriefeat
#print(usefeatdict)
#print("aa")
if __name__ == '__main__':
    featnames = " "
    feat = " "
    u2u1 = " "
    ufu1 = " "
    edges= " "
    keylist1, friendstrlist, alllablelist, arrays_loaded1, arrays_loaded1 = readfile(featnames, feat, u2u1, ufu1, edges)
    selfuselabe,usefriefeat = label_feat(keylist1,friendstrlist,alllablelist,arrays_loaded1, arrays_loaded1)
    with open('labe.pkl','wb') as f7:
        pickle.dump(selfuselabe,f7)
    with open('u2feat.pkl', 'wb') as f6:
        pickle.dump(usefriefeat, f6)