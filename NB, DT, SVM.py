import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


##Naive Bayes###

path = "/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/"
filename = "vectorized_data_wiki.csv"
Wiki_Data = pd.read_csv(path + filename)

Wiki_Data["sentiment"] = Wiki_Data["sentiment"].replace({
    "very positive": "positive",
    "very negative": "negative"
})

Training_Wiki, Testing_Wiki = train_test_split(Wiki_Data, test_size=0.3, random_state=42)

Training_Wiki_Label = Training_Wiki["sentiment"]
Training_Wiki = Training_Wiki.drop(["sentiment"], axis=1)

Testing_Wiki_Label = Testing_Wiki["sentiment"]
Testing_Wiki = Testing_Wiki.drop(["sentiment"], axis=1)

print(Wiki_Data.head())
output_path = "/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/wiki_data_nb.csv"
Wiki_Data.to_csv(output_path, index=False)

print(Training_Wiki.head())
print(Testing_Wiki.head())

print("Train shape:", Training_Wiki.shape)
print("Test shape:", Testing_Wiki.shape)

MyMN = MultinomialNB()
My_MN_Model = MyMN.fit(Training_Wiki, Training_Wiki_Label)
print(My_MN_Model)

Predictions_Wiki = My_MN_Model.predict(Testing_Wiki)
print("Predictions:", Predictions_Wiki)

print("Prediction Probabilities:")
print(My_MN_Model.predict_proba(Testing_Wiki).round(3))

CM_Wiki = confusion_matrix(Testing_Wiki_Label, Predictions_Wiki)
print("Confusion Matrix:\n", CM_Wiki)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_Wiki, display_labels=My_MN_Model.classes_)
plt.title("Multinomial Naive Bayes - Wiki Sentiment")
disp.plot()
plt.show()

print("Accuracy:", accuracy_score(Testing_Wiki_Label, Predictions_Wiki))

## Decision Tree##

# Tree 1: default entropy, max depth = 6
dt1 = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1)
dt1.fit(Training_Wiki, Training_Wiki_Label)
pred1 = dt1.predict(Testing_Wiki)

cm1 = confusion_matrix(Testing_Wiki_Label, pred1, labels=dt1.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=dt1.classes_)
disp1.plot(cmap='Blues')
plt.title("Tree 1: Entropy, Depth = 6")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt1, feature_names=Training_Wiki.columns, class_names=dt1.classes_, filled=True, rounded=True, max_depth=3)
plt.title("Tree 1: Entropy, Depth = 6 (Top Levels)")
plt.show()

# Tree 2: gini split, shallower tree
dt2 = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=2)
dt2.fit(Training_Wiki, Training_Wiki_Label)
pred2 = dt2.predict(Testing_Wiki)

cm2 = confusion_matrix(Testing_Wiki_Label, pred2, labels=dt2.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=dt2.classes_)
disp2.plot(cmap='Blues')
plt.title("Tree 2: Gini, Depth = 4")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt2, feature_names=Training_Wiki.columns, class_names=dt2.classes_, filled=True, rounded=True, max_depth=3)
plt.title("Tree 2: Gini, Depth = 4 (Top Levels)")
plt.show()

# Tree 3: deeper tree with min_samples_split
dt3 = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=10, random_state=3)
dt3.fit(Training_Wiki, Training_Wiki_Label)
pred3 = dt3.predict(Testing_Wiki)

cm3 = confusion_matrix(Testing_Wiki_Label, pred3, labels=dt3.classes_)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=dt3.classes_)
disp3.plot(cmap='Blues')
plt.title("Tree 3: Entropy, Depth = 10, Min Split = 10")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt3, feature_names=Training_Wiki.columns, class_names=dt3.classes_, filled=True, rounded=True, max_depth=3)
plt.title("Tree 3: Entropy, Depth = 10, Min Split = 10 (Top Levels)")
plt.show()

## SVM ##

# Linear SVM
svm_linear = LinearSVC(C=0.1, max_iter=10000, random_state=42)
svm_linear.fit(Training_Wiki, Training_Wiki_Label)
pred_linear = svm_linear.predict(Testing_Wiki)

print("Accuracy:", accuracy_score(Testing_Wiki_Label, pred_linear))

cm_linear = confusion_matrix(Testing_Wiki_Label, pred_linear)
disp_linear = ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=svm_linear.classes_)
disp_linear.plot(cmap='Blues')
plt.title("SVM Linear Kernel (C=0.1)")
plt.show()

# RBF Kernel SVM
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_rbf.fit(Training_Wiki, Training_Wiki_Label)
pred_rbf = svm_rbf.predict(Testing_Wiki)

print("Accuracy:", accuracy_score(Testing_Wiki_Label, pred_rbf))

cm_rbf = confusion_matrix(Testing_Wiki_Label, pred_rbf)
disp_rbf = ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=svm_rbf.classes_)
disp_rbf.plot(cmap='Blues')
plt.title("SVM RBF Kernel (C=1, gamma='scale')")
plt.show()

# Polynomial Kernel SVM
svm_poly = SVC(kernel='poly', degree=3, C=1, gamma='scale', random_state=42)
svm_poly.fit(Training_Wiki, Training_Wiki_Label)
pred_poly = svm_poly.predict(Testing_Wiki)

print("Accuracy:", accuracy_score(Testing_Wiki_Label, pred_poly))

cm_poly = confusion_matrix(Testing_Wiki_Label, pred_poly)
disp_poly = ConfusionMatrixDisplay(confusion_matrix=cm_poly, display_labels=svm_poly.classes_)
disp_poly.plot(cmap='Blues')
plt.title("SVM Polynomial Kernel (degree=3, C=1)")
plt.show()





