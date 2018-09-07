import networkx as nx
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

def get_undirected_graph(file):
    with open(file, 'r') as train:
        G=nx.Graph() # undirected graph
        for line in train:
            neighbour_list=[int(i) for i in line.split()]
            neighbour_tuples=[(neighbour_list[0],neighbour_list[i+1]) for i in range(len(neighbour_list)-1)]
            G.add_edges_from(neighbour_tuples)
        return G
print("Generating undirected graph......")
UG=get_undirected_graph('train.txt')


print("Generating community......")
from networkx.algorithms import community
comms = list(community.asyn_fluidc(UG,100))
print("Size of communities:"+str(len(comms)))

print("Adding community attribute......")
count=0
for node in UG.nodes():
    if(count%10000==0):
        print(count)
    count+=1
    for i in range(len(comms)):
        if node in comms[i]:
            UG.nodes[node]['community'] = i

def generate_positive_features():
    features = []
    count = 0
    print("Generating positive features......")
    for sample in positive_samples:
        if (count % 100 == 0):
            print(count)
        count += 1
        feature = []
        try:
            preds = nx.resource_allocation_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.jaccard_coefficient(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.adamic_adar_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.preferential_attachment(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.cn_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.ra_index_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.within_inter_cluster(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            feature.append(1)  # label=1
            
        except:
            print("one error at: "+str(count))
            pass
        features.append(feature)
    print("positive features: "+str(len(features)))
    return features


def generate_negative_features():
    features = []  
    count = 0
    print("Generating negative features......")
    for sample in negative_samples:
        if (count % 100 == 0):
            print(count)
        count += 1
        feature = []
        try:
            preds = nx.resource_allocation_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.jaccard_coefficient(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.adamic_adar_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.preferential_attachment(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.cn_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.ra_index_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.within_inter_cluster(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            feature.append(0)  # label=0
        except:
            print("one error at: "+str(count))
            pass
        features.append(feature)
        
    print("negative features: "+str(len(features)))
    return features

def generate_test_features():
    features = []
    count = 0
    print("Generating test features......")
    for sample in test_samples:
        if (count % 100 == 0):
            print(count)
        count += 1
        feature = []
        try:
            preds = nx.resource_allocation_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.jaccard_coefficient(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.adamic_adar_index(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.preferential_attachment(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.cn_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.ra_index_soundarajan_hopcroft(UG, [sample])
            for u, v, p in preds:
                feature.append(p)

            preds = nx.within_inter_cluster(UG, [sample])
            for u, v, p in preds:
                feature.append(p)
        except:
            print("one error at: "+str(len(count)))
            pass
        features.append(feature)

    return features

# add features and label, combine
def generate_traning_data():
    positive_features = generate_positive_features()
    negative_features = generate_negative_features()
    features = positive_features + negative_features
    # random.shuffle(features)  
    return features


traning_data = generate_traning_data()

def write_train_to_csv(traning_data):
    with open("train.csv","w",newline="") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(["RA","JC","AA","PA","CSH","RSH","WIC","Label"])
        writer.writerows(traning_data)
        
write_train_to_csv(traning_data)

def write_test_to_csv(test_data):
    with open("test.csv","w",newline="") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(["RA","JC","AA","PA","CSH","RSH","WIC"])
        writer.writerows(test_data)

test_data=generate_test_features()
write_test_to_csv(test_data)




