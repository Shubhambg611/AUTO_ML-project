import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Streamlit UI
st.title("ML Task Selector and Data Preprocessing")
st.write("Upload a CSV file, choose whether to perform regression or classification analysis, and preprocess the data.")

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the total number of rows and columns
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    st.write(f"**Total Rows:** {total_rows}")
    st.write(f"**Total Columns:** {total_columns}")
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Display a preview of the uploaded data in the first column
    with col1:
        st.write("### Data Preview")
        st.dataframe(df, use_container_width=True)  # Expands to full width

    # Display column names and data types in the second column
    with col2:
        st.write("### Column Names and Data Types")
        column_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes
        })
        st.dataframe(column_info)
        
    # User selects target column for analysis
    target_column = st.selectbox("Select the target column for analysis", df.columns)
    
    # User selects the type of task
    task_type = st.radio("Choose the type of task", ("Regression", "Classification"))
    
    # Preprocessing task begins
    st.write("### Preprocessing the data...")

    # Separate features and target
    X = df.copy()  # Keep all columns including the target
    y = df[target_column]

    # Detect categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the target column from feature lists
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Define the preprocessing pipeline
    # 1. Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handling missing values
        ('scaler', StandardScaler())  # Normalizing data
    ])
    
    # 2. Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handling missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
    ])
    
    # 3. Combine numerical and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ], remainder='passthrough'  # Keep other columns unchanged
    )
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Retrieve feature names for numerical and categorical columns after fitting
    numerical_feature_names = numerical_cols
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)

    # Combine all feature names
    all_feature_names = list(numerical_feature_names) + list(ohe_feature_names)

    # Check the shape of preprocessed data and feature names
    st.write(f"Shape of preprocessed data: {X_preprocessed.shape}")
    st.write(f"Number of feature names: {len(all_feature_names)}")

    # Check for column mismatch
    if X_preprocessed.shape[1] != len(all_feature_names):
        st.error("The number of columns in the preprocessed data does not match the feature names.")
        st.write("Preprocessed Data Shape: ", X_preprocessed.shape)
        st.write("Feature Names: ", all_feature_names)
    else:
        # Convert preprocessed data to DataFrame
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_feature_names)
        st.write("### Preprocessed Data:")
        st.dataframe(X_preprocessed_df.head())
    
        st.write("### Summary of Preprocessing Steps:")
        st.write(f"1. **Numerical Columns:** {numerical_feature_names}")
        st.write(f"2. **Categorical Columns:** {categorical_cols}")
        st.write(f"3. **Total Features After Encoding:** {len(all_feature_names)}")
        
        # Display the user's choices
        st.write(f"**Selected Target Column:** {target_column}")
        st.write(f"**Selected Task Type:** {task_type}")
    
        # Note: No further modeling is done, as per your request.
        st.write("You can now proceed to further analysis or modeling as needed with the preprocessed data.")

# If no file is uploaded, display a prompt
else:
    st.write("Please upload a CSV file to start.")
