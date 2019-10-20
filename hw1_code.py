import pandas as pd
import numpy as np
import math
import graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# to load datasets and vectorizer and split into train, test, validate datasets
def load_data():
    #load datasets and verctorizer by using CountVectorizer 
    cv = CountVectorizer()
    df_real = pd.read_table('clean_real.txt', header = None)
    df_real ['Label'] = 1
    df_false = pd.read_table('clean_fake.txt', header = None)
    df_false ['Label'] = 0
    df = df_real.append(df_false)
    df_x = df[0]
    df_y = df['Label']
    x_cv = cv.fit_transform(df_x)
    
    #create train, validation, test datasets
    row_count = x_cv.shape[0]
    indices = list(range(row_count))
    num_training_indices = int(0.7 * row_count)
    num_val_indices = int(0.85 * row_count)
    np.random.shuffle(indices)
    train_indices = indices[:num_training_indices]
    test_indices = indices[num_training_indices+1:num_val_indices]
    val_indices = indices[num_val_indices:]
    x_train, x_test, x_val = x_cv[train_indices], x_cv[test_indices], x_cv[val_indices]
    y_train, y_test, y_val = df_y.iloc[train_indices], df_y.iloc[test_indices], df_y.iloc[val_indices]
    print ('Q1: Load data successfully')
    return df_x, df_y, x_cv, x_train, x_val, x_test, y_train, y_val, y_test, cv


#to select the best model selected from depth 5, 10, 20, 30, 40 
def select_model(x_train, y_train, x_val, y_val):     
    #fit the training datasets to Classifier 
    acc_rate = 0
    depth_test = pd.Series([5,10,20,30,40], index = [0,1,2,3,4])
    cri_test = pd.Series(['gini','entropy'], index = [0,1])
    for a in range (cri_test.count()):
        for b in range (depth_test.count()):
            clf = DecisionTreeClassifier(max_depth=depth_test[b],criterion=cri_test[a])
            clf.fit (x_train, y_train)
            predictions = clf.predict(x_val)
            real = np.array(y_val)
            count = 0
            
            #calculate accurancy of validation data for each combination
            for i in range (len(real)):
                if predictions[i] == real[i]:
                    count =count +1
                    acc_rate1 = count/len(real)
            print('Accurancy is', acc_rate1, 'with decision tree depth equals to', depth_test[b], 'and criterion is', cri_test[a])
            
            # find the classifier with best performance (biggest accurancy) 
            if acc_rate1 >= acc_rate:
                acc_rate = acc_rate1
                depth = depth_test[b]
                cri = cri_test[a]
                classifier = clf
    print('The best classifier has accurancy of', acc_rate, 'with decision tree depth equals to', depth, 'and criterion is', cri)
    return acc_rate, depth, cri, classifier


#function to calculate entropy based on variable p (probability of left/right child)
def entropy(series):
    series_false = series.where(series == 0)
    series_false = series_false.dropna()
    series_false_count = len(series_false)
    series_total_count = len(series)
    p = series_false_count / series_total_count
    H = - (p * math.log2(p)) - ((1-p) * math.log2(1-p))
    return H, series_total_count


def compute_information_gain(feature_name):
    print('Below is the Information Gain calculation for feature_name', feature_name)
    get_features = cv.get_feature_names()
    index = get_features.index(feature_name)
    x_array = x_cv.toarray()
    x_array[:,index]

    # split the root to left and right child
    feature_appear = df_y[x_array[:,index] >= 0.5]
    feature_noappear = df_y[x_array[:,index] < 0.5]
    
    #calculate root entropy
    root_entropy, df_y_total_count = entropy(df_y)
    print('Root entropy is', root_entropy)

    #calculate left child entropy    
    left_entropy, feature_appear_total_count = entropy(feature_appear)
    print('Left child entropy is', left_entropy)

    #calculate right child entropy
    right_entropy, feature_noappear_total_count = entropy(feature_noappear)
    print('Right child entropy is', right_entropy)

    #calculate the information gain of the feature name
    weight = feature_appear_total_count / feature_noappear_total_count
    IG = root_entropy - (weight * left_entropy + (1-weight) * right_entropy) 
    print ('Weight of left child is', weight, 'and right child is', 1-weight)
    print('Information gain of', feature_name, 'is', IG)
    return IG


#function to generate visulization of tree
def generate_tree():
    dot_data = export_graphviz(
        classifier,
        out_file=None,
        feature_names=cv.get_feature_names(),
        class_names=['fake', 'real'],
        filled=True,
        rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph


#outputs
df_x, df_y, x_cv, x_train, x_val, x_test, y_train, y_val, y_test, cv = load_data()
print ( )
print ('Q3: Below are the output of select_model function')
acc_rate, depth, cri, classifier = select_model(x_train, y_train, x_val, y_val)
print ( )
print ('Q4: Below are the output of compute_information_gain function, I use donald as example, pls feel free to use other feature name to calculate')
compute_information_gain('donald')
generate_tree()
