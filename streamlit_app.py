import streamlit as st
import polars as pl
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

st.title("Data Cleansing, Profiling & ML Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pl.read_csv(uploaded_file)
        use_pl = True
        st.success("Data loaded with Polars")
    except Exception:
        df = pd.read_csv(uploaded_file)
        use_pl = False
        st.success("Data loaded with Pandas")

    st.subheader("Raw Data Preview")
    st.write(df.head().to_pandas() if use_pl else df.head())

    if st.button("Generate Profiling Report"):
        prof_df = df.to_pandas() if use_pl else df
        profile = ProfileReport(prof_df, title="Data Profiling Report", explorative=True)
        st.components.v1.html(profile.to_html(), height=600, scrolling=True)

    st.subheader("Data Cleaning Options")
    drop_na = st.checkbox("Drop rows with missing values")
    remove_dupes = st.checkbox("Remove duplicate rows")

    cleaned_df = df
    if drop_na:
        cleaned_df = cleaned_df.drop_nulls() if use_pl else cleaned_df.dropna()
    if remove_dupes:
        cleaned_df = cleaned_df.unique() if use_pl else cleaned_df.drop_duplicates()

    st.write("Preview After Cleaning")
    st.write(cleaned_df.head().to_pandas() if use_pl else cleaned_df.head())

    def convert_df_to_csv(data):
        if use_pl:
            return data.write_csv()
        else:
            return data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Cleaned CSV",
        data=convert_df_to_csv(cleaned_df),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    st.subheader("Simple ML Prediction")

    ml_df = cleaned_df.to_pandas() if use_pl else cleaned_df

    # Option to create a binary target from 'Sold Quantity' column if it exists and user wants
    if 'Sold Quantity' in ml_df.columns:
        create_binary = st.checkbox("Create binary target from 'Sold Quantity' (> 0 means 1, else 0)")
        if create_binary:
            ml_df['binary_target'] = (ml_df['Sold Quantity'] > 0).astype(int)
            target_col = 'binary_target'
        else:
            target_col = st.selectbox("Select target column (binary classification only)", ml_df.columns)
    else:
        target_col = st.selectbox("Select target column (binary classification only)", ml_df.columns)

    if st.checkbox("Train Logistic Regression Model"):
        if ml_df[target_col].nunique() == 2:
            X = ml_df.drop(target_col, axis=1)
            y = ml_df[target_col]

            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = X[col].astype('category').cat.codes

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"Model trained. Accuracy on test set: {accuracy:.2f}")
            joblib.dump(model, 'logistic_model.joblib')
            st.success("Model saved.")
        else:
            st.error("Target column must be binary for logistic regression.")

    if st.checkbox("Make Prediction"):
        try:
            model = joblib.load('logistic_model.joblib')
        except Exception:
            st.error("Train model first to make predictions.")
            model = None

        if model:
            input_text = st.text_area("Enter comma-separated feature values matching training order:")
            if input_text:
                try:
                    input_values = [float(x.strip()) for x in input_text.split(",")]
                    pred = model.predict([input_values])
                    st.write(f"Prediction: {pred[0]}")
                except Exception as e:
                    st.error(f"Invalid input: {e}")
else:
    st.info("Please upload a CSV file to get started.")
