import streamlit as st
import joblib
"""
# Repurchase Prediction
"""

# ['nb_past_orders', 'avg_basket', 'total_purchase_cost', 'avg_quantity','total_quantity', 'avg_nb_unique_products', 'total_nb_codes']
nb_past_orders = st.number_input('number of past order', value=5)
avg_basket = st.number_input('Average basket in $', value=50)
total_purchase_cost = st.number_input('Total purchase cost', value=50)
avg_quantity = st.number_input('average quantity', min_value=1, max_value=100, step=1, value=8)
total_quantity = st.number_input('total quantity', min_value=1, max_value=500, step=1, value=40)
avg_nb_unique_products = st.number_input('average number of unique product', value=20)
total_nb_codes = st.number_input('total number of discount code used', value=2)

value_to_predict=[nb_past_orders, avg_basket, total_purchase_cost, avg_quantity, total_quantity, avg_nb_unique_products, total_nb_codes]

if st.button('Predict'):
    scaler= joblib.load('scaler_prep.joblib')
    value_to_predict=scaler.transform([value_to_predict])
    print(value_to_predict)
    filename = 'model.joblib'
    model = joblib.load(filename)
    pred=model.predict(value_to_predict)[0]
    if pred==1:
        st.write('The customer will repurchase')
    else:
        st.write('The customer will not repurchase')
