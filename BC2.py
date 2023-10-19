from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
# Load the breast cancer dataset
X, y= load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
 
# Train the model
clf = RandomForestClassifier(random_state=23)
clf.fit(X_train, y_train)
 
# preduction
y_pred = clf.predict(X_test)
 
# compute the confusion matrix
cm = confusion_matrix(y_test,y_pred)
#cm = confusion_matrix(actual,predicted)
c_c=["#FFA500","#1E1E1E","#0000FF"]
cmap=sns.color_palette(c_c)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            cmap=cmap,
            fmt='g')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)


