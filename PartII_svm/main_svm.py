import numpy as np
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import metrics
from conf_matrix import func_confusion_matrix

## step 1: load data from csv file. 
data = dl.download_data('crab.csv').values

n = 200
#split data 
S = np.random.permutation(n)
#100 training samples
Xtr = data[S[:100], :6]
Ytr = data[S[:100], 6:]
# 100 testing samples
X_test = data[S[100:], :6]
Y_test = data[S[100:], 6:].ravel()

## step 2 randomly split Xtr/Ytr into two even subsets: use one for training, another for validation.
#############placeholder 1: training/validation #######################
n2 = len(Xtr)
S2 = np.random.permutation(n2)
 
# subsets for training models
x_train = Xtr[S2[:50]]  # First 50 samples for training
y_train = Ytr[S2[:50]].ravel()  # Use ravel() to flatten to 1D array
# subsets for validation
x_validation = Xtr[S2[50:]]  # Last 50 samples for validation
y_validation = Ytr[S2[50:]].ravel()
#############placeholder end #######################

## step 3 Model selection over validation set
# consider the parameters C, kernel types (linear, RBF etc.) and kernel
# parameters if applicable. 


# 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed) 
#  keeping kernel = "linear"
#############placeholder 2: Figure 1#######################
c_range = [0.1, 1.0, 10.0, 100.0]  # Different C values to try on log scale
svm_c_error = []

for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_c_error.append(error)

plt.figure(figsize=(8, 6))
plt.plot(c_range, svm_c_error, 'bo-')  # Blue line with dots
plt.xscale('log')  # Use log scale for C values
plt.title('Linear SVM')
plt.xlabel('C values')
plt.ylabel('Validation Error')
plt.grid(True)
plt.show()
#############placeholder end #######################


# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel ( with other hyperparameters fixed) 
#############placeholder 3: Figure 2#######################
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
best_c = 1.0  # Use the best C value from previous experiment

for kernel_value in kernel_types:
    model = svm.SVC(kernel=kernel_value, C=best_c)
    if kernel_value == 'poly':
        model.set_params(degree=3)  # Set polynomial degree
    elif kernel_value == 'rbf':
        model.set_params(gamma='scale')  # Set RBF parameter
    
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()
#############placeholder end #######################


## step 4 Select the best model and apply it over the testing subset 
#############placeholder 4:testing  #######################

best_kernel = 'poly'
best_c = 1 # poly had many that were the "best"
model = svm.SVC(kernel=best_kernel, C=best_c)
model.fit(X=x_train, y=y_train)

#############placeholder end #######################


## step 5 evaluate your results in terms of accuracy, real, or precision. 

#############placeholder 5: metrics #######################
# func_confusion_matrix is not included
# You might re-use this function for the Part I. 
y_pred = model.predict(X_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y_test, y_pred)

print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

#############placeholder end #######################

#############placeholder 6: success and failure examples #######################
# Success samples: samples for which you model can correctly predict their labels
# Failure samples: samples for which you model can not correctly predict their labels

#############placeholder end #######################



