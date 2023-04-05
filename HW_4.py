import pandas as pd
import numpy as np
from math import log2
from math import trunc
import matplotlib.pyplot as plt
from collections import Counter
import time
from collections import defaultdict

# ________________________________________  Preprocessing ____________________________________-_

# Seperates all data in given DataFrame column (df) into 100 values
def bin(df):
    return pd.cut(df, bins=100, labels=False)


# Bins all numerical value columns, and converts duration_ms -> duration_min
# Purpose: Reduces number of unique instances increasing speed without dropping
# performance too much
def preprocessing(df):
    df['duration_min'] = df['duration_ms'] / 60000
    df = df.drop(columns=['duration_ms'])

    df['speechiness'] = bin(df['speechiness'])
    df['tempo'] = bin(df['tempo'])
    df['valence'] = bin(df['valence'])
    df['liveness'] = bin(df['liveness'])
    df['danceability'] = bin(df['danceability'])
    df['loudness'] = bin(df['loudness'])
    df['acousticness'] = bin(df['acousticness'])
    df['duration_min'] = bin(df['duration_min'])
    df['instrumentalness'] = bin(df['instrumentalness'])
    df['energy'] = bin(df['energy'])

    return df


# _____________________________________ Making Tree Methods ________________________-

# Sums all the values in a counter
# Purpose: Helper method for making dtree
def total(cntr):
    return sum(cntr.values())


# One possible measure of accuracy for a tree
def gini(cntr):
    tot = total(cntr)
    return 1 - sum([(val/tot) ** 2 for val in cntr.values()])


# One possible measure of accuracy for a tree
def entropy(cntr):
    tot = total(cntr)
    return - sum([(v/tot) * log2(v/tot) for v in cntr.values()])


# Helper method to try to nudge values towards converging more
def num_reduction(tot1, tot2, weight):
    return abs(tot1 - tot2) * weight


# Takes the weighted average of the chosen measurement criteria on both
# the counter than meets and does not meat the requirement
def weight_avg(cntr1, cntr2, measure):
    tot1, tot2 = total(cntr1), total(cntr2)
    tot = tot1 + tot2
    num_r = num_reduction(tot1, tot2, 0.0) # Essentially does nothing

    return ((measure(cntr1) * tot1 + measure(cntr2) * tot2) / tot) + num_r


# Finds the best variable and variable number to split within a column for a decision node within a tree
def find_split(split_col, split_var, music_class, df, measure):
    df_split, df_not_split = df[df[split_col] <= split_var], df[df[split_col] > split_var] #Calculates left and right vals

    cntr_split, cntr_not_split = Counter(df_split[music_class]), Counter(df_not_split[music_class])

    return weight_avg(cntr_split, cntr_not_split, measure)


# Finds the best column to split within a dataframe for a decision node within a tree
def find_best_split_col(df, split_col, measure, music_class):
    best_split_var = ('', float('inf'))
    for split_var in set(df[split_col]):
        wavg = find_split(split_col, split_var, music_class, df, measure)
        if wavg < best_split_var[1]:
            best_split_var = (split_var, wavg)

    return best_split_var


# Finds the best decision node split given a DataFrame
def find_best_split(df, measure, music_class):
    best_split = ('', '', float('inf'))
    for split_col in df.columns:
        if not split_col == music_class:
            best_split_var = find_best_split_col(df, split_col, measure, music_class)
            if best_split_var[1] < best_split[2]:
                best_split = (split_col, best_split_var[0], best_split_var[1])

    return best_split


# Recursive, makes the tree by finding the best split, then iterating on the dataframes
# created from the split to find their best split and so on till it hits one of the stopping measures
def make_tree(train, criterion,  depth, max_depth=None, min_instances=2, target_impurity=0.0, print_vals=False):
    cntr = Counter(train['track_genre'])
    maj_class = cntr.most_common(1)[0][0]
    impurity = criterion(cntr)

    if print_vals:
        print(cntr)
        print('Depth: ', depth)
        print('Impurity: ', impurity)
        print('Maj Class: ', maj_class)

    if max(cntr.values()) < min_instances \
            or depth == max_depth \
            or impurity <= target_impurity:

        if print_vals:
            print('Split: None')
            print('_______________________')
        return None, None, total(cntr), maj_class, impurity, depth, None, None

    else:
        best_split = find_best_split(train, criterion, 'track_genre')
        left_vals = train[train[best_split[0]] <= best_split[1]]  #Calculates left values
        right_vals = train[train[best_split[0]] > best_split[1]]  #Calculates right values

        if print_vals:
            print('Split: ', best_split)
            print('Left_vals Shape: ', left_vals.shape)
            print('Right_vals Shape: ', right_vals.shape)
            print('_______________________')

        left, right = None, None
        if left_vals.shape[0] >= 1 and right_vals.shape[0] >= 1:
            left = make_tree(left_vals, criterion, depth + 1, max_depth, min_instances, target_impurity)
            right = make_tree(right_vals, criterion, depth + 1, max_depth, min_instances, target_impurity)

        return best_split[0], best_split[1], total(cntr), maj_class, best_split[2], depth, left, right


# Front facing method for creating a decision tree
def dtree(train, criterion, max_depth=None, min_instances=2, target_impurity=0.0):
    return make_tree(train, criterion, 0, max_depth, min_instances, target_impurity)


# __________________________________  Measuring Accuracy ______________________________________________

# given a single instance of a dataframe (in numpy form) goes through the model to determine what
# class ('pop', 'rock', 'hip-hop') that instance should be categorized into
def walk(tree, val):
    split = (tree[0], tree[1])
    left, right = tree[6], tree[7]
    maj_class = tree[3]

    if split[0] is None and split[1] is None:
        return maj_class

    if val[0][split[0]] <= split[1]: # Calculates left and right value
        if left is None:
            return maj_class
        return walk(left, val)
    else:
        if right is None:
            return maj_class
        return walk(right, val)


# Converts a string of a genre into a number
# Purpose: makes easier to obtain accuracy
#          allows y values to be confusion matrix friendly
def genre_to_num(genre):
    classes = ['pop', 'rock', 'hip-hop']
    return classes.index(genre)


# Helper method for the metrics method
def get_confusion_matrix(y, y_pred):
    unique_classes = set(y) | set(y_pred)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    pred_pair = list(zip(y, y_pred))

    for i, j in pred_pair:
        matrix[int(i), int(j)] += 1

    vertical_sums = np.sum(matrix, axis=0)
    horizontal_sums = np.sum(matrix, axis=1)

    return matrix, vertical_sums, horizontal_sums


# Given a confusion matrix, finds the True/False Positives/Negatives
# for a given class
def read_matrix(matrix, v_sums, h_sums, class_num):
    tp = matrix[class_num, class_num]
    fp = v_sums[class_num] - tp
    fn = h_sums[class_num] - tp
    tn = np.sum(matrix) - (tp + fp + fn)

    return tp, fp, fn, tn


# Given True/False Positive/Negatives calculate the five metrics
# of the class
def calculate_metrics(tp, fp, fn, tn, n):
    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    met = {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Precision': precision,
           'F1': f1}

    return met


# Helper method for averaging a list of five metrics
# dic_list = a dictionary where each key stores a list
def average_metrics(dic_list):
    keys = dic_list.keys()
    length = len(dic_list['Accuracy'])
    mets_avg = {'Accuracy': 0.0, 'Sensitivity': 0.0, 'Specificity': 0.0, 'Precision': 0.0,
                  'F1': 0.0}

    for key in keys:
        mets_avg[key] = sum(dic_list[key]) / length

    return mets_avg


# Calculates five metrics and stores in a dict: Accuracy, Sensitivity, Specificity, Precision, F1
def metrics(y, y_pred):
    n = len(y)
    classes = len(set(y))
    mets_total = {'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'Precision': [],
           'F1': []}

    matrix, v_sums, h_sums = get_confusion_matrix(y, y_pred)

    for class_num in range(classes):
        tp, fp, fn, tn = read_matrix(matrix, v_sums, h_sums, class_num)
        met_class = calculate_metrics(tp, fp, fn, tn, n)

        for key in mets_total.keys():
            mets_total[key].append(met_class[key])

    mets_avg = average_metrics(mets_total)

    return mets_avg


# Splits a dataframe into (folds) number of subsets
# and returns the fth fold
# Calcualtes at runtime so no extra space is used
def k_fold_cross_val(df, f, folds=10):
        train_fold = df[df.index % folds != f]
        valid_fold = df[df.index % folds == f]
        return train_fold, valid_fold


# Returns the model's predictions of the classes for the given data
def predict(model, data):
    preds = []

    for val in range(data.shape[0]):
        preds.append(genre_to_num(walk(model, data.iloc[[val]].to_dict('records'))))

    return np.array(preds)


# Helper method to find the accuracy of a model on a dataset without having
# to run get_overall_accuracy
#
# Front facing method
def get_accuracy(model, data):
    y_pred = predict(model, data)
    y = np.vectorize(genre_to_num)(data['track_genre'].to_numpy())
    mets = metrics(y, y_pred)
    return mets


# Performs k-fold cross validation given a set of hyperparameters
#
# Front facing
def get_overall_accuracy(data, criterion=gini, max_depth=6, min_instances=3, target_impurity=0.23, folds=10):
    mets_total = {'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'Precision': [],
                  'F1': []}

    for k in range(folds):
        train_fold, valid_fold = k_fold_cross_val(data, k, folds)
        model = dtree(train_fold, criterion, max_depth, min_instances, target_impurity)
        #print(model)
        mets = get_accuracy(model, valid_fold)

        print('K: ', k)
        print('Mets: ', mets)

        for key in mets_total.keys():
            mets_total[key].append(mets[key])

    mets_avg = average_metrics(mets_total)
    return mets_avg


# _____________________________________ Hyperparameter Tuning __________________________________________________

# Helper method for graphing a line
def graph_line(x, y, xlabel, ylabel, title, scatterplot=False, colors=None, marker=None):
    plt.figure(figsize=(8, 8))

    if scatterplot:
        plt.scatter(x, y, c=colors)
    else:
        plt.plot(x, y, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Given a hyperparameter and a range you want to test (start, end, steps) the
# parameter on, return the value in the range that achieved highest accuracy
#
# Front facing
def tune_hyp_params(data, hyp_param, param_range, criterion=gini, max_depth=6,
                    min_instances=10, target_impurity=0.23, folds=10,print_graph=True):

    x_vect = list(np.arange(param_range[0], param_range[1], param_range[2]))
    y_vect = []
    dic = {'criterion': criterion, 'max_depth': max_depth, 'min_instances': min_instances, 'target_impurity': target_impurity}
    max_acc = 0
    max_value = 0

    for x in x_vect:
        print("Hyper parameter " + hyp_param + " val: ", x)
        dic[hyp_param] = x
        mets_avg = get_overall_accuracy(data, dic['criterion'], dic['max_depth'], dic['min_instances'],
                                        dic['target_impurity'], folds)
        avg_acc = mets_avg['Accuracy']
        y_vect.append(avg_acc)
        print("Avg Accuracy at " + hyp_param + " value " + str(x) + ": ", avg_acc)

        if avg_acc > max_acc:
            max_acc = avg_acc
            max_value = x

    if print_graph:
        graph_line(x_vect, y_vect, xlabel= hyp_param + " values", ylabel="Avg Accuracy",
                   title="Accuracy Over Values for " + hyp_param, marker='o')

    print("Max Accuracy: ", max_acc)
    print("Value: ", max_value)
    return max_value


# Main:

df_test = pd.read_csv('/Users/mihir/DS4400 - Spring ML 1/Homework 4/spotify_test.csv')
df_train = df = pd.read_csv('/Users/mihir/DS4400 - Spring ML 1/Homework 4/spotify_train.csv')

# Preprocess datasets
df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

# Shuffle
df_train = df_train.sample(frac=1)
df_test = df_test.sample(frac=1)

start_time = time.time() # for reducing time

# K-Fold accuracy of dataframe (just to know)
print(get_overall_accuracy(df_train))

# Make the model
model = dtree(df_train, criterion=gini, max_depth=6, min_instances=3, target_impurity=0.23)

# Get the test accuracy
print(get_accuracy(model, df_test))

# Stop the clock
print('Execution Time in Seconds: ', time.time() - start_time)