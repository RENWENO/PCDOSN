#! -*- coding: utf-8 -*-


def DataStriping(u2u1,ufu1,uau1,hyper,u2feat,lable,rate):
    train=[]
    test=[]
    keylist1 = []
    for key1 in u2u1.keys():
        keylist1.append(key1)
    num = round(len(keylist1)*rate)
    trainx=keylist1[:num]
    testx = keylist1[num:]

    print(train)
    trainu2u1,trainufu1,trainuau1,trainhyper,trainu2feat,trainlable={},{},{},{},{},{}
    for tk in trainx:
        trainu2u1[tk] = u2u1[tk]
        trainufu1[tk] = ufu1[tk]
        trainuau1[tk] = uau1[tk]
        trainhyper[tk]= hyper[tk]
        trainu2feat[tk] = u2feat[tk]
        trainlable[tk]  = lable[tk]
    train.append(trainu2u1)
    train.append(trainufu1)
    train.append( trainuau1)
    train.append(trainhyper)
    train.append(trainu2feat)
    train.append(trainlable)

    testu2u1,testufu1,testuau1,testhyper,testu2feat,testlable={},{},{},{},{},{}
    for ts in testx:
        testu2u1[ts] = u2u1[ts]
        testufu1[ts] = ufu1[ts]
        testuau1[ts] = uau1[ts]
        testhyper[ts] = hyper[ts]
        testu2feat[ts] = u2feat[ts]
        testlable[ts] = lable[ts]

    test.append(testu2u1)
    test.append(testufu1)
    test.append(testuau1)
    test.append(testhyper)
    test.append(testu2feat)
    test.append(testlable)
    return train,test

