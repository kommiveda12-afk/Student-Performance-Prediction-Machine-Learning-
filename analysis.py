

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# load cleaned data
df=pd.read_csv("venv/data/student_cleaned_data.csv")
print(df.head())
# stats report
print(df.columns)
print('\nStatiscal Summary ')
imp_columns=[
    'study_time','failures','absences','exam1_score','exam2_score',
    'exam3_score']
print(df[imp_columns].describe())
# calculate a final column
df["final_score"]=(df["exam1_score"]+df["exam2_score"]+df["exam3_score"])/3
print(df.head())
# score distribution (to check count of good performance students )
sns.histplot(df["final_score"],bins=20,kde=True)
plt.title("Distribution of Final Score")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.savefig("final_score_distribution.png")
plt.show()
print("Figure 1:Graph Explanation\n")
print("Most students scored between 9 and 14 marks.")
print("The distribution looks approximately normal (bell-shaped).")
print("Very few students scored extremely low or extremely high.")
# creating a clustering patten here
print("\n")
print("the performance grading to the students")
df["Performance"]=df["final_score"].apply(
    lambda x:'Low' if x<8 else 'Medium'if x<12 else 'High'
)
print(df[['final_score','Performance']].head())
print("\n")
sns.countplot(x='Performance',data=df)
plt.title("Student Performance CAtegories ")
plt.xlabel("Performance Level")
plt.ylabel("Number of Studnets")
plt.savefig("peformance_categories.png")
plt.show()
print("The majority of teh studenst fall into the medium Performance category ,while Fewer students fall into low and high category")
# coversion of categorical columns
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['Performance'] = df['Performance'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})
print("\nEncoded Dataset Preview\n")
print(df[['sex','Performance']].head())
# checking for teh values
print("\nUnique values after encoding")
print("sex:", df['sex'].unique())
print("Performance:", df['Performance'].unique())

# ml train and predict
x=df.drop(['Performance','student_id','final_score'],axis=1)
y=df['Performance']
print("Features used for training:\n",x.columns)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)
print("Training dat size:",x_train.shape)
print("Testing data size:",x_test.shape)

# decision tree
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

# prediction
y_pred=model.predict(x_test)
print("Sample Predictions:"),y_pred[:10]
print("Accuracy\n")
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy:",accuracy)
# factors that influence the most
feature_importance = pd.Series(
    model.feature_importances_,index=x.columns).sort_values(ascending=False)
# graph
feature_importance.plot(kind='bar')
plt.title("Feature Importance in Predicting Student Performance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.savefig("feature_importance.png")
plt.show()
print("Feature Importance:\n")
print(feature_importance)
# confuion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)
# visual
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig("confusion_matrix.png")
plt.show()
print("matrix output explanation\n")
print("The confusion matrix shows that most predictions fall along the diagonal,\n"
      "indicating that the model correctly classifies the majority of students"
      "into their performance categories.\n"
      " Only a few misclassifications occur between Medium and High categories,\n"
      " showing that the model performs highly accurately.")

