import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


string = "Machine Learning"
st.set_page_config(page_title=string, page_icon="😃")

st.title("Machine Learning")
st.write("""
# Salary Prediction Model
Salary vs. *Experience*
""")

df = pd.read_csv("Salary_Data.csv")

X = df.iloc[:,[0]].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

exp = st.sidebar.slider('Experience',1,10,2)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict([[exp]])

st.write(f"Experience: ", exp)
st.write(f"Salary: ", float(y_pred))

st.write("""
# Scatter Plot
Salary vs. *Experience*
""")

fig = plt.figure()
plt.scatter(X, y, alpha=0.8, cmap='viridis')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()

st.pyplot(fig)