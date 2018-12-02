from config import *
import numpy as np

def get_one_hot(label, dimE):
    return (np.arange(dimE)==label).astype(np.integer)


def static(trainset, testset):
    alldata = np.append(trainset,testset)
    edgeset = {}
    maxlen = -1
    for data in alldata:
        if len(data['tokens']) > maxlen:
            maxlen = len(data['tokens'])
        for edge in data['edgeSet']:
            if edge['kbID'] not in edgeset:
                edgeset[edge['kbID']] = len(edgeset)
    edgeset['others'] = len(edgeset)
    return [edgeset, maxlen]
        

def posequal(pos1, pos2):
    if(len(pos1) != len(pos2)):
        return False
    for i in range(len(pos1)):
        if(pos1[i] != pos2[i]):
            return False
    return True

def prepare(dataset, edgeset, maxlen, name):
    print("Prepare data set %s" % name)
    X = []
    y = []
    for m in range(len(dataset)):
        data = dataset[m]
        if m % 10000 == 0:
            print("Prepare data %d in dataset %s" % (m, name))
        for i in range(len(data['vertexSet'])):
            for j in range(len(data['vertexSet'])):
                left = data['vertexSet'][i]
                right = data['vertexSet'][j]
                traindata = dict()
                traindata['left'] = left['tokenpositions']
                traindata['right'] = right['tokenpositions']
                traindata['token'] = np.append(data['tokens'], ['' for ii in range(maxlen - len(data['tokens']))])
                X.append(traindata)
                for edge in data['edgeSet']:
                    if left['tokenpositions'] == edge['left'] and right['tokenpositions'] == edge['right']:
                        y.append(edgeset[edge['kbID']])
                    else:
                        y.append(edgeset['others'])
    np.save("../datahot/" + name + "_X.npy", X)
    np.save("../datahot/" + name + "_y.npy", y)

def load_data():
    print("Load raw data...")

    print("Load train set raw...")
    train_raw = np.load(FLAGS.dataset_path + FLAGS.trainset_name)
    print("Load test set raw...")
    test_raw = np.load(FLAGS.dataset_path + FLAGS.testset_name)

    print("Prepare edgeset...")
    [edgeset,maxlen] = static(train_raw, test_raw)
    print("Maxlen = %d" % maxlen)

    prepare(train_raw, edgeset, maxlen, "train")
    prepare(test_raw, edgeset, maxlen, "test")


if __name__ == '__main__':
    load_data()
