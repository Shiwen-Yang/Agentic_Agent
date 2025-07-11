import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from scipy.stats import skew
import statsmodels.api as sm

st.set_page_config(layout="wide")

st.title("🧮 Interactive Linear Regression Builder")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")

    # Show header with option to load more
    st.dataframe(df.head())
    if st.checkbox("Show full data"):
        st.dataframe(df)

    # Filter box
    st.subheader("🔍 Filter Criteria")
    filter_expr = st.text_input("Enter filter expression (e.g., Age > 30 & Gender == 'Male')")
    if filter_expr:
        try:
            df = df.query(filter_expr)
        except Exception as e:
            st.error(f"Invalid filter expression: {e}")

    # Data summary
    st.subheader("📊 Data Summary")
    numeric_df = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        'variable name': numeric_df.columns,
        '% NA': df[numeric_df.columns].isna().mean().values * 100,
        'min': numeric_df.min().values,
        '25%': numeric_df.quantile(0.25).values,
        'median': numeric_df.median().values,
        '75%': numeric_df.quantile(0.75).values,
        'max': numeric_df.max().values,
        'std dev.': numeric_df.std().values,
        'skewness': numeric_df.skew().values
    })
    st.dataframe(summary)

    # Correlation heatmap
    st.subheader("📈 Correlation Matrix")
    drop_cols = st.multiselect("Drop columns (e.g., IDs)", df.columns.tolist())
    corr_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    corr_matrix = corr_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Data Visualization
    st.subheader("📉 Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Univariate Distribution")
        var = st.selectbox("Select variable", df.columns)
        bin_width = st.slider("Adjust bin width", 1, 100, 10)
        fig, ax = plt.subplots()
        if df[var].dtype in [np.int64, np.float64]:
            sns.histplot(df[var].dropna(), kde=True, bins=bin_width, ax=ax)
        else:
            df[var].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("### Bivariate Plot")
        x_var = st.selectbox("X-axis", df.columns, key='x')
        y_options = [col for col in df.columns if col != x_var]
        y_var = st.selectbox("Y-axis", y_options, key='y')

        sampled_df = df[[x_var, y_var]].dropna()
        if len(sampled_df) > 500:
            sampled_df = sampled_df.sample(500)
        fig, ax = plt.subplots()
        sns.scatterplot(data=sampled_df, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)

    # Linear Regression Section
    st.subheader("📐 Build Linear Regression Model")

    # Select response variable
    response = st.selectbox("Select response variable", numeric_df.columns, key='response')

    # Select explanatory variables
    explanatory_candidates = [col for col in numeric_df.columns if col != response]
    X = numeric_df[explanatory_candidates].dropna()
    y = numeric_df[response].loc[X.index]

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    p_values = model.pvalues.drop("const")

    significant_vars = p_values[p_values < 0.05].index.tolist()

    selected_explanatory = st.multiselect(
        "Select explanatory variables",
        options=explanatory_candidates,
        default=significant_vars
    )

    if selected_explanatory:
        X_final = sm.add_constant(numeric_df[selected_explanatory].dropna())
        y_final = numeric_df[response].loc[X_final.index]
        final_model = sm.OLS(y_final, X_final).fit()

        st.write("### 📋 Regression Coefficient Table")
        coef_df = pd.DataFrame({
            "Variable": final_model.params.index,
            "Coefficient": final_model.params.values,
            "Std. Error": final_model.bse.values,
            "t-Statistic": final_model.tvalues.values,
            "p-Value": final_model.pvalues.values
        }).reset_index(drop=True)

        st.dataframe(coef_df.style.format({
            "Coefficient": "{:.4f}",
            "Std. Error": "{:.4f}",
            "t-Statistic": "{:.2f}",
            "p-Value": "{:.4f}"
        }))

        st.markdown("### 📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared", f"{final_model.rsquared:.4f}")
        col2.metric("Adj. R-squared", f"{final_model.rsquared_adj:.4f}")
        col3.metric("F-statistic", f"{final_model.fvalue:.2f}")

        # Residual Plot
        st.markdown("### 🧪 Residual Plot")
        fitted_vals = final_model.fittedvalues
        residuals = final_model.resid
        fig, ax = plt.subplots()
        sns.scatterplot(x=fitted_vals, y=residuals, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted Values")
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")