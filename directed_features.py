from scipy.sparse import csr_matrix
import csv
import numpy as np
print("Loading samples......")
positive_samples=np.load("positive_samples.npy").tolist()
negative_samples=np.load("negative_samples.npy").tolist()
def get_test_samples():
    with open('test-public.txt') as test:
        test_edges = []
        for line in test:
            edge_list = line.split()
            try:
                test_edges.append((int(edge_list[1]), int(edge_list[2])))
            except:
                pass
        return test_edges
test_samples = get_test_samples()

########################################################################################################
print("Generating matrix......")
with open('train.txt','r') as train:
    row=[]
    col=[]
    data=[]
    for line in train:
        neighbour_list=[int(i) for i in line.split()]
        for i in range(1,len(neighbour_list)):
            row.append(neighbour_list[0])
            col.append(neighbour_list[i])
            data.append(1)
    source_sink_matrix=csr_matrix((data,(row,col)),shape=(4867136,4867136))
    sink_source_matrix=csr_matrix((data,(col,row)),shape=(4867136,4867136)) # row slicing is fater than column slicing in csr_matrix
################################### Feature method ###################################################
import numpy as np
from math import sqrt

def cos_sim(X,Y):
    try:
        return (np.dot(X,Y.T)/(sqrt(X.nnz)*sqrt(Y.nnz))).toarray()[0][0]
    except:
        return 0

def suc_size(key):
    return source_sink_matrix[key].nnz
def pre_size(key):
    return sink_source_matrix[key].nnz
def pre_pre_set(key1,key2):
    return len(set(sink_source_matrix[key1].nonzero()[1])&set(sink_source_matrix[key2].nonzero()[1]))
def suc_pre_set(key1,key2):
    return len(set(source_sink_matrix[key1].nonzero()[1])&set(sink_source_matrix[key2].nonzero()[1]))
def pre_pre_cos(key1,key2):
    return cos_sim(sink_source_matrix[key1],sink_source_matrix[key2])
def suc_pre_cos(key1,key2):
    return cos_sim(source_sink_matrix[key1],sink_source_matrix[key2])
def wuyifan(key1,key2):
    feature=[0]*100
    i=0
    for key in source_sink_matrix[key1].nonzero()[1]:
        feature[i]=(cos_sim(sink_source_matrix[key],sink_source_matrix[key2]))
        i+=1
        if (i >= 100):
            break
    feature.sort(reverse=True)
    return feature

######################################## Generating features #############################################################
def generate_positive_features():
    features = []
    count = 0
    print("Generating positive features......")
    for sample in positive_samples:
        if (count % 10 == 0):
            print(count)
        count += 1
        feature = []
        try:
            feature.append(suc_size(sample[0]))
            feature.append(suc_size(sample[1]))
            feature.append(pre_size(sample[0]))
            feature.append(pre_size(sample[1]))
            feature.append(pre_pre_set(sample[0],sample[1]))
            feature.append(suc_pre_set(sample[0], sample[1]))
            feature.append(pre_pre_cos(sample[0], sample[1]))
            feature.append(suc_pre_cos(sample[0], sample[1]))
            feature.extend(wuyifan(sample[0], sample[1]))
            feature.append(1)  # label=1

        except:
            print("one error at: " + str(count))
            pass
        features.append(feature)
    print("positive features: " + str(len(features)))
    return features

def generate_negative_features():
    features = []
    count = 0
    print("Generating negative features......")
    for sample in negative_samples:
        if (count % 10 == 0):
            print(count)
        count += 1
        feature = []
        try:
            feature.append(suc_size(sample[0]))
            feature.append(suc_size(sample[1]))
            feature.append(pre_size(sample[0]))
            feature.append(pre_size(sample[1]))
            feature.append(pre_pre_set(sample[0],sample[1]))
            feature.append(suc_pre_set(sample[0], sample[1]))
            feature.append(pre_pre_cos(sample[0], sample[1]))
            feature.append(suc_pre_cos(sample[0], sample[1]))
            feature.extend(wuyifan(sample[0], sample[1]))
            feature.append(0)  # label=0
        except:
            print("one error at: " + str(count))
            pass
        features.append(feature)
    print("negative features: " + str(len(features)))
    return features

def generate_test_features():
    features = []
    count = 0
    print("Generating test features......")
    for sample in test_samples:
        if (count % 10 == 0):
            print(count)
        count += 1
        feature = []
        try:
            feature.append(suc_size(sample[0]))
            feature.append(suc_size(sample[1]))
            feature.append(pre_size(sample[0]))
            feature.append(pre_size(sample[1]))
            feature.append(pre_pre_set(sample[0],sample[1]))
            feature.append(suc_pre_set(sample[0], sample[1]))
            feature.append(pre_pre_cos(sample[0], sample[1]))
            feature.append(suc_pre_cos(sample[0], sample[1]))
            feature.extend(wuyifan(sample[0], sample[1]))
        except:
            print("one error at: " + str(count))
            pass
        features.append(feature)
    print("test features: " + str(len(features)))
    return features
##############################################################################################################
def generate_traning_data():
    positive_features = generate_positive_features()
    negative_features = generate_negative_features()
    features = positive_features + negative_features
    # random.shuffle(features)
    return features

traning_data = generate_traning_data()

def write_train_to_csv(traning_data):
    with open("train_directed.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["RA", "JC", "AA", "PA", "CSH", "RSH", "WIC", "Label"])
        writer.writerows(traning_data)
write_train_to_csv(traning_data)

def write_test_to_csv(test_data):
    with open("test_directed.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["SA", "SB", "PA", "PPS", "SPS", "PPC", "SPC","WYF"])
        writer.writerows(test_data)
test_data = generate_test_features()
write_test_to_csv(test_data)



