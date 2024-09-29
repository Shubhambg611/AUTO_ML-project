import streamlit as st
import pandas as pd

# Streamlit UI
st.title("ML Task Selector: Regression or Classification?")
st.write("Upload a CSV file and choose whether to perform regression or classification analysis.")

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
        st.dataframe(df.head())

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
    
    # Display the user's choices
    st.write(f"**Selected Target Column:** {target_column}")
    st.write(f"**Selected Task Type:** {task_type}")
    
    # Note: No further processing or modeling is done, as per your request.
    st.write("You can now proceed to further analysis or modeling as needed.")

# If no file is uploaded, display a prompt
else:
    st.write("Please upload a CSV file to start.")
