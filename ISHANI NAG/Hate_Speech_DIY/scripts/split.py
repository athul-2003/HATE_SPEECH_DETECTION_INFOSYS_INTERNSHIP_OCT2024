import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer  # Import for text vectorization

# Step 1: Load your original dataset
data = pd.read_csv(r'C:\Users\dell\Desktop\Infosys Springboard Project\Hate_Speech_DIY\data\hate_speech_data.csv')

# Create a binary target column based on 'hate_speech'
data['target_column'] = data['hate_speech'].apply(lambda x: 1 if x > 0 else 0)  # Map values: 1 if hate_speech > 0, else 0

# Preprocess Text Data - Convert tweets into numerical features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit the number of features for efficiency
tweet_features = vectorizer.fit_transform(data['tweet']).toarray()  # Transform tweets into numerical features

# Combine vectorized features with other non-text columns
features = pd.DataFrame(tweet_features)  # Convert sparse matrix to DataFrame
features = pd.concat([features, data.drop(columns=['tweet', 'hate_speech', 'target_column'], errors='ignore')], axis=1)

# Convert all column names to strings to avoid type mismatch issues
features.columns = features.columns.astype(str)

# Separate features and target
X = features
y = data['target_column']

# Step 2: Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and targets into a new DataFrame
smote_data = pd.DataFrame(X_resampled, columns=features.columns)
smote_data['target_column'] = y_resampled  # Append the target variable back

# Step 3: Delete existing train/test CSV files if they exist
files_to_delete = [
    'train_features.csv',
    'train_target.csv',
    'test_features.csv',
    'test_target.csv'
]

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")
    else:
        print(f"{file} does not exist.")

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    smote_data.drop('target_column', axis=1),
    smote_data['target_column'],
    test_size=0.2, 
    random_state=42,
    stratify=smote_data['target_column']
)

# Save splits to CSVs
X_train.to_csv('train_features.csv', index=False)
y_train.to_csv('train_target.csv', index=False)
X_test.to_csv('test_features.csv', index=False)
y_test.to_csv('test_target.csv', index=False)

print("Training and testing datasets saved successfully!")
