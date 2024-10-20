# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Trained Models
@st.cache_resource
def load_models():
    sales_model = joblib.load('random_forest_sales_model.pkl')
    price_model = lgb.Booster(model_file='trained_lightgbm_model.txt')
    cv = joblib.load('cv.pkl')
    tv = joblib.load('tv.pkl')
    lb = joblib.load('lb.pkl')
    return sales_model, price_model, cv, tv, lb

sales_model, price_model, cv, tv, lb = load_models()

st.title("Store Sales and Price Prediction")

# Step 3: Get user input for Store Features
st.subheader("Store Features")
store = st.number_input("Store", value=2)
customers = st.number_input("Customers", value=624)
competition_distance = st.number_input("Competition Distance", value=570)
promo = st.selectbox("Promo", [0, 1], index=1)
promo2 = st.selectbox("Promo2", [0, 1], index=1)
state_holiday = st.selectbox("State Holiday", [0, 1], index=0)
store_type = st.selectbox("Store Type", [0, 1], index=0)
assortment = st.selectbox("Assortment", [0, 1], index=0)
day_of_week = st.number_input("Day of the Week", value=5)
week = st.number_input("Week", value=5)
day = st.number_input("Day", value=30)
month = st.number_input("Month", value=1)
year = st.number_input("Year", value=2015)

# Step 3: Prepare Store Features DataFrame
store_df = pd.DataFrame({
    'Store': [store],
    'Customers': [customers],
    'CompetitionDistance': [competition_distance],
    'Promo': [promo],
    'Promo2': [promo2],
    'StateHoliday': [state_holiday],
    'StoreType': [store_type],
    'Assortment': [assortment],
    'DayOfWeek': [day_of_week],
    'Week': [week],
    'Day': [day],
    'Month': [month],
    'Year': [year],
    'AvgSales': [5919],
    'AvgCustomers': [624],
    'AvgSalesPerCustomer': [9.49],
    'MedSales': [5919],
    'MedCustomers': [624],
    'MedSalesPerCustomer': [9.49],
    'CompetitionOpenSinceMonth': [11],
    'CompetitionOpenSinceYear': [2007],
    'Promo2SinceWeek': [13],
    'Promo2SinceYear': [2010]
})

# Predict store sales
predicted_store_sales = sales_model.predict(store_df)[0]
st.write(f"Predicted Store Sales: {predicted_store_sales}")

# Step 4: Product Features Input
st.subheader("Product Features")
product_name = st.text_input("Product Name", "Vintage Designer Bag")
item_condition_id = st.selectbox("Item Condition", [1, 2, 3, 4, 5], index=2)
category_name = st.text_input("Category", "Women/Bags/Handbags")
brand_name = st.text_input("Brand", "Gucci")
shipping = st.selectbox("Shipping", [0, 1], index=1)
item_description = st.text_area("Item Description", "A luxury handbag in great condition")

# Prepare Product Features DataFrame
product_df = pd.DataFrame({
    'name': [product_name],
    'item_condition_id': [item_condition_id],
    'category_name': [category_name],
    'brand_name': [brand_name],
    'shipping': [shipping],
    'item_description': [item_description]
})

# Preprocess and Vectorize
def handle_missing_inplace(dataset):
    if 'missing' not in dataset['category_name'].cat.categories:
        dataset['category_name'] = dataset['category_name'].cat.add_categories('missing')
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].replace('No description yet', 'missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

handle_missing_inplace(product_df)
to_categorical(product_df)

X_name_input = cv.transform(product_df['name'])
X_category_input = cv.transform(product_df['category_name'])
X_description_input = tv.transform(product_df['item_description'])
X_brand_input = lb.transform(product_df['brand_name'])
X_dummies_input = csr_matrix(pd.get_dummies(product_df[['item_condition_id', 'shipping']], sparse=True).values)

sparse_input = hstack((X_dummies_input, X_description_input, X_brand_input, X_category_input, X_name_input)).tocsr()

# Predict price
predicted_price = price_model.predict(sparse_input, num_iteration=price_model.best_iteration, predict_disable_shape_check=True)
base_price = np.expm1(predicted_price)[0]

# Adjusted Price based on store sales performance
if predicted_store_sales > 20000:
    adjusted_price = base_price * 1.25
elif 15000 < predicted_store_sales <= 20000:
    adjusted_price = base_price * 1.15
elif 10000 < predicted_store_sales <= 15000:
    adjusted_price = base_price * 1.05
elif 7000 < predicted_store_sales <= 10000:
    adjusted_price = base_price
elif 4000 < predicted_store_sales <= 7000:
    adjusted_price = base_price * 0.90
else:
    adjusted_price = base_price * 0.80

st.write(f"Base Product Price: {base_price:.2f}")
st.write(f"Adjusted Product Price: {adjusted_price:.2f}")

