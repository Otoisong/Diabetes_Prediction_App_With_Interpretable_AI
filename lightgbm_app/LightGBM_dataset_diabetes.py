import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import lightgbm as lgb  # LightGBM library
import joblib

#################################
# Import Sample Data
#################################

# Bring in the saved CSV file
my_df = pd.read_csv("diabetes_prediction_dataset.csv")
my_df.head()


# Shuffle data
#my_df = shuffle(my_df, random_state=42)

''' CLASS BALANCE '''
# Check for class balance
my_df["diabetes"].value_counts()

###################################
# Dealing with missing value
####################################

# Check for missing values
my_df.isna().sum()

#Shape
my_df.shape

# Drop any missing values
my_df.dropna(how="any", inplace=True)

##### Split Input Variable and Output Variable ######
my_df.columns
X = my_df[['gender', 'age', 'hypertension','heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level']]
#X = my_df.iloc[:, :-1]  # Input features

y = my_df.iloc[:, -1]   # Output variable (target)

### Split the data into Training and Testing sets ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Map 'gender' and 'smoking_history' to numeric values directly, without using OneHotEncoder
# Assuming 'gender' has two values: Male, Female
# Assuming 'smoking_history' has three values: never, formerly, current

# Define mappings
gender_mapping = {"Male": 1, "Female": 0}
smoking_history_mapping = {"never": 0, "formerly": 1, "current": 2}

# Apply mappings to the training and test sets
X_train['gender'] = X_train['gender'].map(gender_mapping)
X_test['gender'] = X_test['gender'].map(gender_mapping)

X_train['smoking_history'] = X_train['smoking_history'].map(smoking_history_mapping)
X_test['smoking_history'] = X_test['smoking_history'].map(smoking_history_mapping)

################################################
# Model Training with XGBoost
################################################

# Create an instance of the LightGBM classifier
clf = lgb.LGBMClassifier()

# Train the model
clf.fit(X_train, y_train)

###############################################
# Model Assessment
###############################################

# Predict the classes for the test set
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class

''' Confusion Matrix '''
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Plot confusion matrix
plt.style.use("seaborn-v0_8-poster")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha="center", va="center", fontsize=20)
plt.show()

# Model evaluation metrics
print("LightGBM_diabetes_data_Kaggle_Result")
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Precision:", precision_score(y_test, y_pred_class))
print("Recall:", recall_score(y_test, y_pred_class))
print("F1 Score:", f1_score(y_test, y_pred_class))

##################################################
# Feature Importance (XGBoost built-in importance)
##################################################

# LightGBM provides feature importances, so we extract and visualize them
feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', inplace=True)

# Plot feature importance
plt.barh(feature_importances.index, feature_importances['importance'])
plt.title("Feature Importance - LightGBM")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

################################################
# Permutation Importance
################################################

# Permutation importance is more reliable and model-agnostic
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

permutation_importances = pd.DataFrame(result["importances_mean"], index=X_train.columns, columns=['permutation_importance'])
permutation_importances.sort_values(by='permutation_importance', inplace=True)

# Plot permutation importance
plt.barh(permutation_importances.index, permutation_importances['permutation_importance'])
plt.title("Permutation Importance - XGBoost")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()


# Save the model to a file
joblib.dump(clf, 'lightgbm_Saved_model.pkl')




# Load the model
loaded_model = joblib.load('lightgbm_Saved_model.pkl')

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)






