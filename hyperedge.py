#! -*- coding: utf-8 -*-
import pickle
import re
import numpy as np
#from transformers import BertModel, BertTokenizer

def re1(String):
    ori =  String
    cleaned_string1 = re.sub(";anonymized", '', ori)
    cleaned_string2 = re.sub(";id", '', cleaned_string1)
    cleaned_string = re.sub(";", ' ', cleaned_string2)
    return  cleaned_string

def readfile(featnames,edges,feat,PKLfile):
#Gets the attribute dictionary and feature dictionary, as well as the attribute type
    with open(featnames, "r", encoding="utf-8") as f4:
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
            Fvelue = feat[-2] + " " + feat[-1][:-1]
            featdict[num] = Fvelue
        Atttiedict = {}
        for i, velu in enumerate(Attributed):
            Atttiedict[i] = velu
        #
        #
        print(attributedict)
        print(featdict)
        print(Attributed)

    # The user's friend structure relationship
    with open(edges, "r", encoding="utf-8") as f2:
        edgs = f2.readlines()
        use = set()
        friendlist = []
        for ed in edgs:
            edi = ed[:-1].split(" ")
            use.add(edi[0])
            # use.add(edi[1])
        use_frient = {}
        for u in use:
            usefrident = set()
            for edd in edgs:
                if u == edd[:-1].split(" ")[0]:
                    usefrident.add(edd[:-1].split(" ")[1])

            friendlist.append(usefrident)
            use_frient[u] = usefrident
            if u == "1584":
                print(len(usefrident))
            if u == "1909":
                print(len(usefrident))

    # Get all user characteristics
    with open(feat, 'r', encoding="utf-8") as f5:
        allusefeat = f5.readlines()
        usefeatdict = {}
        for usefeat in allusefeat:
            AttriV = np.zeros((1, 23))
            uf = usefeat[:-1].split(" ")
            key = uf[0]
            veul1 = uf[1:]
            veul = np.array([int(char) for char in uf[1:]])
            for j, V in enumerate(velu):
                if V == 1:
                    S = attributedict[j]
                    for k, v in Atttiedict:
                        if v == S:
                            key = k
            # attrie = np.
            usefeatdict[key] = veul
        print(usefeatdict)

    allusebian = {}
    for k in usefeatdict.keys():
        print(k)
        arr = usefeatdict[k]
        bian = np.hstack((arr[89:219], arr[266:284], arr[350:375], arr[384:442]))
        allusebian[k] = bian

    print(allusebian)
    with open(PKLfile, 'rb') as f1:
        arrays_loaded1 = pickle.load(f1)
        keylist1 = []
        for key1 in arrays_loaded1.keys():
            keylist1.append(key1)
    return keylist1,use_frient,allusebian

def hyprtedge(keylist1,use_frient,allusebian):
    hyperedge = {}
    for kl in keylist1:
        print(kl)
        akl = use_frient[kl]
        arrlist = []
        for uf in akl:
            arr = allusebian[uf]
            arrlist.append(arr)
        al = np.array(arrlist)
        chao = np.dot(al, al.T)
        hyperedge[kl] = chao
    return hyperedge


if __name__ == '__main__':
    #Enter a behavior feature name
    featnames = " "
    #Enter all user relationship file paths
    edges = " "
    # Enter the path of all user behavior attribute feature files
    feat = " "
    #User to user structure
    PKLfile =" "
    keylist1,use_frient,allusebian = readfile(featnames,edges,feat,PKLfile)

    hyperedge = hyprtedge(keylist1,use_frient,allusebian)
    with open('hyperedge.pkl', 'wb') as f6:
        pickle.dump(hyperedge, f6)

