import numpy as np
import pandas as pd




file_path =r"D:\downloads\Customer-Churn-Records.csv"

def load_file():
    df = pd.read_csv(file_path)
    print("data loaded successfully")
    #print(df.columns)
    #print(df.isnull().sum())    #no missing values



load_file()

df = pd.read_csv(file_path)

df['Loyalty_Score'] = df['Tenure'] / df['Age']
df['Financial_Stability'] = df['Balance'] / df['EstimatedSalary']
df['Credit_Risk'] = df['CreditScore'] / df['Age']
df['Engagement_Score'] = df['NumOfProducts'] * df['IsActiveMember']
df['Loyalty_Indicator'] = df['Point Earned'] * df['Satisfaction Score']


#print(df.head())
import numpy as np

# Function to detect outliers using IQR
def detect_outliers_iqr(df, features):
    outlier_counts = {}
    
    for feature in features:
        Q1 = df[feature].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[feature].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        
        lower_bound = Q1 - 1.5 * IQR  # Lower bound
        upper_bound = Q3 + 1.5 * IQR  # Upper bound
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)

        #print(f"{feature}: {len(outliers)} outliers detected")
    
    return outlier_counts

# Select numerical features for outlier detection
features_to_check = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 
                     'Point Earned', 'Loyalty_Score', 'Financial_Stability', 
                     'Credit_Risk', 'Loyalty_Indicator']

# Detect outliers
outlier_results = detect_outliers_iqr(df, features_to_check)
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use("ggplot")

# Plot boxplots for each numerical feature
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(3, 3, i)  # Create subplots (3 rows, 3 columns)
    sns.boxplot(y=df[feature])
    plt.title(feature)

plt.tight_layout()  # Adjust layout for better readability
#plt.show()

# Function to remove extreme outliers & cap moderate ones
def handle_outliers(df, features_to_remove, features_to_cap):
    # Remove extreme outliers
    for feature in features_to_remove:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    
    # Cap moderate outliers
    for feature in features_to_cap:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
        df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    
    return df

# Features to remove extreme outliers from
features_to_remove = ['Age']

# Features to cap outliers
features_to_cap = ['Financial_Stability', 'Credit_Risk', 'Loyalty_Score']

# Apply the function
df_cleaned = handle_outliers(df, features_to_remove, features_to_cap)

#print(f"New dataset size after outlier handling: {df_cleaned.shape}")


from sklearn.preprocessing import StandardScaler

# Define features to scale
features_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 
                     'Point Earned', 'Loyalty_Score', 'Financial_Stability', 
                     'Credit_Risk', 'Loyalty_Indicator']

# Initialize StandardScaler
scaler = StandardScaler()

# Apply scaling
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Display the scaled dataframe
#print(df.head())



df = pd.get_dummies(df, columns=['Geography', 'Card Type'], drop_first=True)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

#print(df.head())  # View first 5 rows
#print(df.columns)
#print(df.dtypes)



# Define X (features) and y (target variable)
X = df.drop(columns=['Exited' , 'RowNumber' , 'CustomerId' , 'Surname'])  # Drop target column
y = df['Exited']  # Target variable (1 = churned, 0 = stayed)

#check class imbalance

import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of churned vs. non-churned customers
print(y.value_counts(normalize=True) * 100)  # Shows percentage distribution
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Make churn 33%
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)  # Fit & transform training set
X_test_scaled = scaler.transform(X_test)  # Only transform test set (no fitting!)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_smote_scaled, y_train_smote)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
