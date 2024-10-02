#! -*- coding: utf-8 -*-
import re
import pickle
import numpy as np


def re1(String):
    ori = String
    cleaned_string1 = re.sub(";anonymized", '', ori)
    cleaned_string2 = re.sub(";id", '', cleaned_string1)
    cleaned_string = re.sub(";", ' ', cleaned_string2)
    return cleaned_string


def readfile(egofeat,edges,featdir,featnames):
    with open(egofeat, "r", encoding="utf-8") as f:
        egofeat = f.readlines()
        # print(egofeat)
        # print(len(egofeat[0].split(" ")))
    #The user's friend structure relationship
    with open(edges, "r", encoding="utf-8") as f2:
        edgs = f2.readlines()
        use = set()
        friendlist = []
        for ed in edgs:
            edi = ed[:-1].split(" ")
            use.add(edi[0])
            # use.add(edi[1])
        for u in use:
            usefrident = set()
            for edd in edgs:
                if u == edd[:-1].split(" ")[0]:
                    usefrident.add(edd[:-1].split(" ")[1])
                # if u== edd[:-1].split(" ")[1]:
                #     usefrident.add(edd[:-1].split(" ")[0])
            friendlist.append(usefrident)
            if u == "1584":
                print(len(usefrident))
            if u == "1909":
                print(len(usefrident))

    use = list(use)
    list1 = []
    for i, numfriend in enumerate(friendlist):
        print(len(numfriend))

        if len(numfriend) < 20:
            list1.append(i)
    for i in sorted(list1, reverse=True):  # Delete from back to forward to avoid index offset
        del use[i]
        del friendlist[i]

    # user feature
    with open(featdir, "r", encoding="utf-8") as f3:
        featname = f3.readlines()
        feat1 = featname[1][:-1].split(" ")
        print(len(feat1))
        print(feat1)


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

    #Get all user feature
    with open(featdir, "r", encoding="utf-8") as f5:
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
        # print(usefeatdict)

    return use,egofeat,edgs,feat1,friendlist,Atttiedict,featdict,attributedict,Atttiedict,usefeatdict

def friend_feat_graph(friendlist,usefeatdict):
    allfriendfeat = []
    for friend in friendlist:
        friendAlist = []
        for fri in friend:
            arr = usefeatdict[fri]
            print(arr)
            friendAlist.append(arr)
        friedarr = np.array(friendAlist)
        allfriendfeat.append(friedarr)
    return allfriendfeat

#Analyze the friend structure relationship u-u in the user friend circle
def friend_graph(friendlist,edgs):
    print("开始计算")
    ii= 0
    allself_friend_grapt=[]
    for fri in friendlist:
        print("calculate"+str(ii))
        leng = len(fri)
        #Create an array
        friendmax = np.zeros((leng,leng))
        for i, fri1 in enumerate(fri):
            for j,fri2 in enumerate(fri):
                frient = fri1+" "+fri2+"\n"
                if frient in edgs:
                    friendmax[i][j]=1.0
        allself_friend_grapt.append(friendmax)
        ii = ii+1

    return allself_friend_grapt



def use_use_attri(use_friend,friend_attri,A2A):

    allu2u2A = []
    for i,g in enumerate(use_friend):
        g2 = np.copy(g)
        allu2u2A.append(g2)
    return allu2u2A

#Computes the user-attribute feature-user-attribute diagram
def use_feat_use_attri(use_friend,friend_attri,A2A):
    allU2F2U2A =[]
    for i,f in enumerate(friend_attri):
        g1 = np.copy(use_friend[i])
        for j in range(f.shape[0]):
            for x in range(j+1,f.shape[0]):
                if np.sum(f[j,:]*f[x,:]) > 0:
                    if g1[j,x] == 1:
                        print("A relationship exists")
                    else:
                        g1[j,x]=1.0
                        print("Add Relation")

        allU2F2U2A.append(g1)
    return allU2F2U2A

#Computing Users -- Attribute features -- Attribute features -- Users
def use_feat_feat_use_attri(use_friend,friend_attri,A2A):
    allU2F2U2A =[]
    alla2a = []
    #Gets the coordinates for all correlations
    for i in range(A2A.shape[0]):
        for j in range(A2A.shape[1]):
            if A2A[i, j] == 1:
                #If the element is equal to 1, its coordinates are added to the list
                alla2a.append((i, j))

    for i,f in enumerate(friend_attri):

        g3=np.copy(use_friend[i])
        for a2a in alla2a:
            a1 = a2a[0]
            a2 = a2a[1]
            temp = f[:,a1]+f[:,a2]
            f = np.concatenate((f,temp[:,np.newaxis]), axis=1)
        for j in range(f.shape[0]):
            for x in range(j+1,f.shape[0]):
                if np.sum(f[j,:]*f[x,:]) > 0:
                    g3[j,x]=1
        allU2F2U2A.append(g3)
        print("num"+str(i)+"use")
    return allU2F2U2A

def compute_transitive_closure(adj_matrix):
    n = adj_matrix.shape[0]
    transitive_closure = np.eye(n, dtype=int)

    power = adj_matrix
    while np.any(power):
        #transitive_closure += power  # 累加到传递矩阵
        transitive_closure = transitive_closure + power
        power = np.matmul(power, adj_matrix)

    return transitive_closure
#Computing Users -- Attribute features -- Attribute features -- Users
def useG_useG_attri(u2u,ufu,uaau):
    allU2F2U2A = []
    for i in range(len(u2u)):
        u2u1 = compute_transitive_closure(u2u[i])
        ufu1 = compute_transitive_closure(ufu[i])
        uaau1 = compute_transitive_closure(uaau[i])
        U2U = u2u1+ufu1+uaau1
        allU2F2U2A.append(U2U)
    return allU2F2U2A


if __name__ == '__main__':
    #Enter the path of the user behavior signature file
    useAB = " "
    #Enter all user relationship file paths
    userelation = " "
    #Enter the path of all user behavior attribute feature files
    alluseab = " "
    #Enter a behavior feature name
    ABName = " "
    use, egofeat, edgs, feat1, friendlist, Atttiedict, featdict, attributedict, Atttiedict, usefeatdict = readfile(
        useAB , userelation, alluseab, ABName)
    print("Calculate the friend graph")
    use_friend = friend_graph(friendlist, edgs)
    print("Calculate friend properties")
    friend_attri = friend_feat_graph(friendlist, usefeatdict)
    print("Get attribute relation")
    A2A = np.load('shuxinguanxi.npy')
    u2u=use_use_attri(use_friend,friend_attri,A2A)
    # #np.save('u2u.npy', u2u)
    ufu=use_feat_use_attri(use_friend,friend_attri,A2A)
    #np.save('ufu.npy',ufu)
    uaau = use_feat_feat_use_attri(use_friend,friend_attri,A2A)

    #utu = useG_useG_attri(u2u,ufu,uaau)
    u2udict = {}
    ufudict = {}
    uaaudict = {}
    #utudict = {}
    for u in range(len(use)):
        key = use[u]
        velu1 = u2u[u]
        velu2 = ufu[u]
        velu3 = uaau[u]
        #velu4 = utu[u]
        u2udict[key] = velu1
        ufudict[key] = velu2
        uaaudict[key] = velu3
        #utudict[key] = velu4
    with open('u2u1.pkl', 'wb') as f6:
        pickle.dump(u2udict, f6)

    with open('ufu1.pkl', 'wb') as f7:
        pickle.dump(ufudict, f7)

    with open('uaau1.pkl', 'wb') as f8:
        pickle.dump(uaaudict, f8)

#with open('utu1.pkl', 'wb') as f9:
    #pickle.dump(utudict, f9)
print("。。")