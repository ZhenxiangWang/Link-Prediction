import numpy as np

print("Loading train data......")
train_data={}
with open('train.txt','r') as train:
    for line in train:
        neighbour_list=[int(i) for i in line.split()]
        train_data[neighbour_list[0]]=set([neighbour_list[i+1] for i in range(len(neighbour_list)-1)])

def get_train_sources_and_sinks(file):
    with open(file, 'r') as train:
        sources=set()
        sinks=set()
        for line in train:
            neighbour_list=[int(i) for i in line.split()]
            sources.add(neighbour_list[0])
            for i in range(len(neighbour_list)-1):
                sinks.add(neighbour_list[i+1])
        return sources,sinks

train_sources,train_sinks=get_train_sources_and_sinks('train.txt')
print(len(train_sources))
print(len(train_sinks))

import random

def positive_sampling():
    print("Positive sampling......")
    positive_samples=[]
    count=0
    for i in range(51100):
        if (count % 1000 == 0):
            print(count)
        count+=1
        source_random_index=random.randint(0,19999)
        source=(list(train_sources))[source_random_index]
        origin_sinks=train_data[source] # origin_sinks is a set
        try:
            sink=random.choice(list(origin_sinks))
            positive_samples.append((source,sink))
        except:
            # print(origin_sinks)
            pass
    print(len(positive_samples))
    return positive_samples


def negative_sampling():
    print("Negative sampling......")
    negative_samples=[]
    count = 0
    for i in range(50020):
        if (count % 10 == 0):
            print(count)
        count+=1
        source_random_index=random.randint(0,19999)
        source=(list(train_sources))[source_random_index]
        origin_sinks = train_data[source]
        sink=random.choice(list(train_sinks))
        if sink not in origin_sinks:
            negative_samples.append((source, sink))
    print(len(negative_samples))
    return negative_samples


positive_samples=positive_sampling()
np.save('positive_samples.npy',np.array(positive_samples))

negative_samples=negative_sampling()
np.save('negative_samples.npy',np.array(negative_samples))
