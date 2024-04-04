import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Assuming your CSV file is named 'calc_participant_participation_order.csv' and is in the same directory as your Python script
file_path = "C:\\Users\\ASUS VIVOBOOK\\Desktop\\\אוניברסיטה\\תשפד\\תואר שני\\למידה עמוקה\\פרוייקט\calc_participant_participation_order.csv "

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Filter the DataFrame to include only rows where the Id starts with 'B'
df_filtered = df[df['Id'].str.startswith('B')]

nan_counts = df_filtered.isna().sum()

# Drop the specified columns
columns_to_drop = ["If you chose someone else, please specify", "Semantic", "TTC", "Latency"]
df_cleaned = df_filtered.drop(columns=columns_to_drop, axis=1)
df_cleaned = df_cleaned  .dropna(subset=['BarrierPassingVehicles'])
df_cleaned = df_cleaned.dropna(thresh=df.shape[1]-20)
nan_counts = df_cleaned.isna().sum()

unique_incomes = df_cleaned ['Income'].unique()
most_common_value = df_cleaned['Income'].mode()[0]
df_cleaned['Income'].fillna(most_common_value, inplace=True)


# List to store column names with NaN values
columns_with_nan = []

# Iterate over columns and check for NaN values
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().any():
        columns_with_nan.append(col)

# Create a new DataFrame with 'Id' column and columns with NaN values
df_nan_columns = df_cleaned[['Id'] + columns_with_nan]
# Columns to fill NaN values with median
columns_to_fill = ['PupilDiameter.max', 'PupilDiameter.mean', 'PupilDiameter.quantile.85', 'PupilDiameter.std']
# Calculate median for each column
medians = df_cleaned[columns_to_fill].median()
# Fill NaN values with median for each column
df_cleaned[columns_to_fill] = df_cleaned[columns_to_fill].fillna(medians)
# Calculate the median 'Informative' value for each group defined by 'Id'
median_informative =df_cleaned.groupby('Id')['Informative'].transform('median')

# Fill the missing values in 'Informative' column with the calculated median values
df_cleaned['Informative'] = df_cleaned['Informative'].fillna(median_informative)


# Calculate the most common value for each group defined by 'Id' in 'CommonFace'
most_common_face_per_id = df_cleaned.groupby('Id')['CommonFace'].agg(lambda x: x.value_counts().idxmax())

# Fill NaN values in 'CommonFace' column with the most common value per group
df_cleaned['CommonFace'] = df_cleaned['CommonFace'].fillna(df_cleaned['Id'].map(most_common_face_per_id))

# Calculate the most common value for each group defined by 'Id' in 'CommonColor'
most_common_color_per_id = df_cleaned.groupby('Id')['CommonColor'].agg(lambda x: x.value_counts().idxmax())

# Fill NaN values in 'CommonColor' column with the most common value per group
df_cleaned['CommonColor'] = df_cleaned['CommonColor'].fillna(df_cleaned['Id'].map(most_common_color_per_id))

# Calculate the median value for each group defined by 'Id' in 'Mood' and 'Fatigue'
median_mood_per_id = df_cleaned.groupby('Id')['Mood'].transform('median')
median_fatigue_per_id = df_cleaned.groupby('Id')['Fatigue'].transform('median')

# Fill NaN values in 'Mood' and 'Fatigue' columns with the median values per group
df_cleaned['Mood'] = df_cleaned['Mood'].fillna(median_mood_per_id)
df_cleaned['Fatigue'] = df_cleaned['Fatigue'].fillna(median_fatigue_per_id)


# Assuming df_cleaned is your DataFrame
# Fill NaN values with median for specified columns
column_to_fill=['DiscussProblems','ComplexThings', 'TimesOfNeed', 'Depend','FeelDeepDown', 'OpeningUp', 'Abandon', 'Care', 'Worry', 'Trust']
medians = df_cleaned[column_to_fill].median()
# Fill NaN values with median for each column
df_cleaned[column_to_fill] = df_cleaned[column_to_fill].fillna(medians)


# Assuming df_cleaned is your DataFrame
# Fill NaN values in the "Relationship" column with the mode
most_common_relationship = df_cleaned['Relationship'].mode()[0]
df_cleaned['Relationship'] = df_cleaned['Relationship'].fillna(most_common_relationship)
# Find unique values in the 'experience computer games' column from df_cleaned
unique_values = df_cleaned['experience computer games'].unique()
df_cleaned['experience computer games'] = df_cleaned['experience computer games'].replace('אין', 'לא', regex=True)
# Replace "באקס בוקס" and "Call of duty" with "כן" in the 'experience computer games' column
df_cleaned['experience computer games'] = df_cleaned['experience computer games'].replace(['באקס בוקס', 'Call of duty'], 'כן', regex=True)
# Replace "באקס בוקס " and "Call of duty " (with trailing space) with "כן" in the 'experience computer games' column
df_cleaned['experience computer games'] = df_cleaned['experience computer games'].replace(['באקס בוקס ', 'Call of duty '], 'כן', regex=True)
df_cleaned['experience computer games'] = df_cleaned['experience computer games'].str.strip()

# Now the replacements have been made
print(df_cleaned['experience computer games'].unique())  # This will print the unique values in the column after replacement




# Count NaN values in each column
nan_count_per_column = df_cleaned.isna().sum()

# Get columns with NaN values
columns_with_nan = nan_count_per_column[nan_count_per_column > 0].index.tolist()

# Total number of NaN values
total_nan_count = nan_count_per_column.sum()

print("Total number of NaN values:", total_nan_count)
print("Columns with NaN values:", columns_with_nan)


# List of columns for feature data
feature_columns = [ 'Scenario', 'Condition', 'Duration', 'WithinIDOrder', 'Distance_Driven', 'collision', 'ExtreemBrakingEvents', 'Mental Demand', 'Physical Demand', 'Temporal Demand', 'Effort','Frustration', 'Performance (reverse)', 'ECG_Rate.mean', 'ECG_Rate.std', 'EDA_Phasic.mean', 'EDA_Phasic.std', 'HRV_HF.mean', 'HRV_HF.std', 'HRV_SDNN.mean', 'HRV_SDNN.std', 'PupilDiameter.mean', 'PupilDiameter.std', 'SCR_Events.mean', 'SCR_Events.std',  'Age', 'Gender', 'experience simulator', 'license years', 'experience computer games',  'accident non-keeping distance', 'technologies life easier',  'understand technology', 'enjoy understanding technology', 'experience simulator', 'FamilyStatus', 'Income', 'NegativeFeedbackEvents', 'Informative', 'CommonFace', 'CommonColor', 'Mood', 'Fatigue', 'PreferredInterface', 'EffectiveInterfaceForSafeDriving', 'DiscussProblems', 'ComplexThings', 'TimesOfNeed', 'Depend', 'FeelDeepDown', 'OpeningUp', 'Abandon', 'Care', 'Worry', 'Trust', 'Relationship', 'BarrierPassingVehicles']

# Select the feature data from df_cleaned
feature_data = df_cleaned[feature_columns]


# Columns to normalize
columns_to_normalize = ['ECG_Rate.mean', 'ECG_Rate.std', 'EDA_Phasic.mean', 'EDA_Phasic.std','HRV_HF.mean', 'HRV_HF.std', 'HRV_SDNN.mean', 'HRV_SDNN.std','PupilDiameter.mean', 'PupilDiameter.std', 'SCR_Events.mean', 'SCR_Events.std','Duration', 'WithinIDOrder', 'Distance_Driven', 'ExtreemBrakingEvents', 'Age', 'license years', 'NegativeFeedbackEvents', 'Mood', 'Fatigue']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the selected columns
feature_data [columns_to_normalize] = scaler.fit_transform(feature_data [columns_to_normalize])



# Columns to normalize
likert_columns = ['Mental Demand', 'Physical Demand', 'Temporal Demand', 'Effort', 'Frustration', 'Performance (reverse)','technologies life easier','understand technology','enjoy understanding technology', 'Informative','DiscussProblems',	'ComplexThings'	,'TimesOfNeed',	'Depend'	,'FeelDeepDown'	,'OpeningUp',	'Abandon',	'Care'	,'Worry',	'Trust']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
feature_data[likert_columns] = scaler.fit_transform(feature_data[likert_columns])

# List of categorical columns for one-hot encoding
categorical_columns = ['Scenario', 'Condition', 'Gender', 'experience simulator','experience computer games', 'accident non-keeping distance','FamilyStatus', 'Income', 'CommonFace', 'CommonColor','PreferredInterface', 'EffectiveInterfaceForSafeDriving', 'Relationship']

# Perform one-hot encoding for categorical columns
feature_data_encoded = pd.get_dummies(feature_data, columns=categorical_columns)

# Now feature_data_encoded contains one-hot encoded columns for the categorical variables
  # This will print the first few rows of the DataFrame
############################################################################################################################################################################
# Split the data into features (X) and target variable (y)
X = feature_data_encoded.drop(columns=['BarrierPassingVehicles'])  # Features
y = feature_data_encoded['BarrierPassingVehicles']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Define the number of folds for cross-validation
num_folds = 4

# Initialize lists to store the evaluation results for each fold
train_scores = []
test_scores = []

# Initialize a KFold cross-validation splitter
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate over each fold
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for this fold
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Build and train the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train_fold.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=16, verbose=0)  # Training
    
    # Evaluate the model on training and testing sets for this fold
    train_pred = model.predict(X_train_fold)
    test_pred = model.predict(X_test_fold)
    
    # Compute and store the evaluation metrics for this fold
    train_rmse = np.sqrt(mean_squared_error(y_train_fold, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_fold, test_pred))
    train_scores.append(train_rmse)
    test_scores.append(test_rmse)

# Compute the average evaluation metrics across all folds
avg_train_rmse = np.mean(train_scores)
avg_test_rmse = np.mean(test_scores)


#

# Define the Random Forest and Gradient Boosting models
random_forest_model = RandomForestRegressor(n_estimators=50, random_state=26)
gradient_boosting_model = GradientBoostingRegressor(n_estimators=50, random_state=26)

# Train the Random Forest and Gradient Boosting models
# Train the Random Forest and Gradient Boosting models
random_forest_model.fit(X_train, y_train)
gradient_boosting_model.fit(X_train, y_train)

# Evaluate the models on the testing set
rf_test_pred = random_forest_model.predict(X_test)
gb_test_pred = gradient_boosting_model.predict(X_test)

# Compute RMSE for each model on the testing set
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))

print("Random Forest Test RMSE:", rf_test_rmse)
print("Gradient Boosting Test RMSE:", gb_test_rmse)
print("cross val avg Train RMSE:", avg_train_rmse)
print("cross val Test avg RMSE:", avg_test_rmse)

######################################################################################################################33

