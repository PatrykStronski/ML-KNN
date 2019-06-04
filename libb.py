import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

threshold = 0.0 

def learn():
    classified,k = model_selection_knn(x_test_a, x_train_a, y_test_a, y_train_a, [1,2,3,4,5,6,7])
    classes = []
    good = 0
    for i in range(0,classified.shape[0]):
        classes.append(np.argmax(classified[i]))
        if(classes[i]==y_test_a[i]):
            good+=1
    return good/len(classified),k,threshold

def hamming_distance(x,x_train):
    x = np.array(x)
    x_train = np.array(x_train)
    x[x<threshold]=0
    x_train[x_train<threshold]=0
    return np.sum(abs(x-x_train))


def encapsulate_h_d(xes,xes_train):
    diffs = np.empty((len(xes),len(xes_train)))
    for x in range(0,len(xes)):
        for xt in range(0,len(xes_train)):
            diffs[x][xt]=hamming_distance(xes[x],xes_train[xt])
    return diffs

def sort_train_labels_knn(Dist, y):
    output = []
    for row in Dist:
        output.append([x for _,x in sorted(zip(row,y))])
    output = np.array(output)
    return output

def calculate_prob(y,ys,k):
    ys = np.array(ys[:k])
    unique, cont = np.unique(ys,return_counts=True)
    counts = dict(zip(unique,cont))
    if(y in counts.keys()):
        return counts[y]/k
    else:
        return 0

def p_y_x_knn(y, k):
    output = np.empty((len(y),10))
    for r in range(0,len(y)):
        for i in range(0,10):
            output[r][i] = (calculate_prob(i, y[r], k))
    return output

def classify(x,k,tr):
    threshold = tr
    model = pkl.load(open("model.pkl","rb"))
    probabsMatrix = p_y_x_knn(sort_train_labels_knn(encapsulate_h_d(x, model[0]), model[1]), k)
    out = []
    for p in probabsMatrix:
        out.append(p.argmax())
    return out


