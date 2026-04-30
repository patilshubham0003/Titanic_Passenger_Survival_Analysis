import pickle as pkl
import numpy as np
import streamlit as st

with open("model.pkl","rb") as f:
    model=pkl.load(f)
    
with open("oneHotEncoder.pkl","rb") as f:
    OHE = pkl.load(f)    
    
st.set_page_config(
    page_title="Titanic Insight",
    page_icon="./4d07142d-67b5-4be9-818f-3cb1ed9b4690.png",
    layout="centered"
)    
    
st.title("Titanic Passenger Survival Analysis")    

Pclass = st.selectbox("Passenger Class",[1,2,3],format_func= lambda x:{1:"first class",2:"second class",3:"third class"}[x])
Sex = st.selectbox("Sex",[0,1],format_func=lambda x:"female" if x==0 else "Male")
Age = st.number_input("Age",0,120,25)
SibSP = st.number_input("Number of siblings",0,10,1)
Parch = st.number_input("Number of parents",0,6,1)
Fare = st.number_input("Ticket price paid by the passenger",0,600,100)
Embarked = st.selectbox("Port where the passenger boarded the Titanic(Embarked)",["C","Q","S"],format_func=lambda x:{"C":"Cherbourg","Q":"Queenstown","S":"Southampton"}[x])

OHE_data=OHE.transform([[Embarked]])[0]
input_arr = np.array([Pclass,Sex,Age,SibSP,Parch,Fare])

final_input = np.concatenate([input_arr,OHE_data]).reshape(1, -1)

if st.button("Predict"):
    output = model.predict(final_input)
    
    if output[0] == 0:
        st.success("Not Survived")
    else: 
        st.success("Survived")