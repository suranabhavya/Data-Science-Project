import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.under_sampling import RandomUnderSampler

pad_bool = pd.read_csv('B:\BOSTON UNI\Acad\TDS\datasets\PAD_cleaned_bool.csv')

Y = pad_bool['violation_bool']

X = pad_bool.drop(columns=['violation_bool', 'PID', 'GIS_ID', 'BLDG_SEQ'])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# rus = RandomUnderSampler(random_state=42)
# X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

# print("After undersampling:", np.bincount(y_train_under))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
# model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))