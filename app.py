import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("shopping_behavior_updated.csv")

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
genders = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
locations = st.sidebar.multiselect("Select Location", options=df["Location"].unique(), default=df["Location"].unique())
categories = st.sidebar.multiselect("Select Category", options=df["Category"].unique(), default=df["Category"].unique())

filtered_df = df[(df["Gender"].isin(genders)) &
                 (df["Location"].isin(locations)) &
                 (df["Category"].isin(categories))]

# Dashboard Title
st.title("🛍️ Shopping Behavior Analytics Dashboard")
st.markdown("Analyze customer behavior across different shopping attributes.")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", filtered_df["Customer ID"].nunique())
col2.metric("Average Purchase (USD)", f"${filtered_df['Purchase Amount (USD)'].mean():.2f}")
col3.metric("Average Review Rating", f"{filtered_df['Review Rating'].mean():.2f} ⭐")

# Chart: Purchase Amount by Category
st.subheader("📊 Purchase Amount by Category")
category_sales = filtered_df.groupby("Category")["Purchase Amount (USD)"].sum().reset_index()
fig1 = px.bar(category_sales, x="Category", y="Purchase Amount (USD)", color="Category",
              title="Total Purchase Amount by Category")
st.plotly_chart(fig1)

# Chart: Gender Distribution
st.subheader("🧍 Gender Distribution")
fig2 = px.pie(filtered_df, names="Gender", title="Gender Breakdown")
st.plotly_chart(fig2)

# Chart: Age Distribution
st.subheader("🎂 Age Distribution")
fig3 = px.histogram(filtered_df, x="Age", nbins=20, title="Customer Age Distribution")
st.plotly_chart(fig3)

# Box Plot: Purchase Amount by Category
st.subheader("💸 Purchase Amount by Category")
fig4 = px.box(filtered_df, x="Category", y="Purchase Amount (USD)", color="Category",
              title="Spending Patterns by Category")
st.plotly_chart(fig4)

# Data Table
st.subheader("📋 Customer Data Table")
st.dataframe(filtered_df.reset_index(drop=True))
