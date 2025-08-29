import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from tqdm import tqdm


def data_save(X_train_list, y_train_list, X_test_list, y_test_list):
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    np.save('./Numpy/X_train.npy', X_train)
    np.save('./Numpy/y_train.npy', y_train)
    np.save('./Numpy/X_test.npy', X_test)
    np.save('./Numpy/y_test.npy', y_test)


def svm():
    X_train = np.load('./Numpy/X_train.npy')
    y_train = np.load('./Numpy/y_train.npy')
    X_test = np.load('./Numpy/X_test.npy')
    y_test = np.load('./Numpy/y_test.npy')

    # print(len(X_train))
    # print(len(y_train))
    # print(len(X_test))
    # print(len(y_test))

    svm = SVC(probability=True)

    param_grid = {
        'C': [0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': [0.01, 0.1]
    }
    grid_search = GridSearchCV(svm, param_grid, cv=5)

    for params in tqdm(list(param_grid.keys()), desc='Grid Search Progress'):
        grid_search.fit(X_train, y_train)
        print(f"Parameters: {params}, Best Score: {grid_search.best_score_}, Best Params: {grid_search.best_params_}")

    # with open('./Numpy/best_svm_model.pkl', 'rb') as file:
    #     best_svm = pickle.load(file)

    best_svm = grid_search.best_estimator_

    y_score = best_svm.decision_function(X_test)

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(lb.classes_))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(lb.classes_)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(lb.classes_)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

    for i in range(len(lb.classes_)):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    # title = 'xx'
    # plt.title(title, fontsize=20)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    best_svm = grid_search.best_estimator_

    with open('./Numpy/best_svm_model.pkl', 'wb') as file:
        pickle.dump(best_svm, file)

    # with open('./data/best_svm_model.pkl', 'rb') as file:
    #     best_svm = pickle.load(file)

    y_pred = best_svm.predict(X_test)

    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion)
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    thresh = confusion.max() / 2
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, confusion[i, j], horizontalalignment="center", verticalalignment="center", fontsize=22,
                 color="white" if confusion[i, j] > thresh else "black")
    # plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()
    report = classification_report(y_test, y_pred, zero_division=1, digits=4)
    print("Classification Report:")
    print(report)
if __name__ == '__main__':
    svm()
