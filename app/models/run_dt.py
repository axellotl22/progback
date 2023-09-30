from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from decision_tree_model import DecisionTree, SplitStrategy
def convert_text_to_categorical(df):
    """
    Convert all text columns in a DataFrame to categorical columns.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - DataFrame with text columns converted to categorical columns
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # if column has text values
            df[col] = df[col].astype('category').cat.codes  # convert to categorical codes
    return df
def extract_feature_names(df, target_column):
    """
    Extrahiert die Feature-Namen aus einem DataFrame.

    Parameters:
    - df: pandas DataFrame
    - target_column: Name der Ziel-Spalte (z.B. "Drug")

    Returns:
    - Liste der Feature-Namen
    """
    return [col for col in df.columns if col != target_column]

#file_path='/home/ubuntu/programmierprojekt/progback/test/Employee.csv'
#file_path='/home/ubuntu/programmierprojekt/progback/test/drug200.csv'
#df = pd.read_csv(file_path)
#df = convert_text_to_categorical(df)
#target_column=("Drug")
#target_column="Education"

# Features und Labels extrahieren
#X = df.drop(target_column, axis=1).values  # Target Column Name = Zu klassifizierende Spalte
#y = df[target_column].values

# Feature-Namen extrahieren
#feature_names = df.columns[:-1].tolist()


data = datasets.load_digits()
#data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(split_strategy=SplitStrategy.DURCHSCHNITT)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy (y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)



acc = accuracy(y_test, predictions)
print(acc)
json_structure = clf.to_json()
print(json_structure)
#feature_names = extract_feature_names(df, "Drug")
print("\n")
#print(feature_names)
print("Confusion Matrix:")
print(clf.confusion_matrix(y_test, predictions))