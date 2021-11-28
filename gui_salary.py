import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time
import pickle

st.title("Salary Classification App")
st.text("Deploy Machine Learning Model using Streamlit,Docker and Minikube Kubernetes Cluster")

if st.checkbox("Show Dataset"):
    st.dataframe(pd.read_csv("salary.csv"))

if st.checkbox("Show Code"):
   st.text('''
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy

# Load dataset
df = pd.read_csv("salary.csv")

# filling missing values

col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", numpy.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation','relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)


X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf_gini, open("model.pkl","wb"))
'''
)

st.header("Please enter the required fields : ")
df = pd.read_csv("salary.csv")

age = st.slider("Enter Age : ",int(df.age.min()),int(df.age.max()),int(df.age.mean()))

w_class = st.selectbox("Working Class : ",["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"])

if w_class == "Federal-gov":
    w_class = 0  
if w_class == "Local-gov":
    w_class = 1
if w_class == "Never-worked":
    w_class = 2
if w_class == "Private":
    w_class = 3
if w_class == "Self-emp-inc":
    w_class =4
if w_class == "Self-emp-not-inc":
    w_class = 5
if w_class == "State-gov":
    w_class = 6
if w_class == "Without-pay":
    w_class = 7



edu = st.selectbox("Education : ", ["10th","11th","12th","1st-4th","Bachelors","Doctorate","Masters","Preschool"])

if edu == "10th":
    edu = 0
if edu == "11th":
    edu = 1
if edu == "12th":
    edu = 2 
if edu == "1st-4th":
    edu = 3
if edu == "Bachelors":
    edu = 9
if edu == "Doctorate":
    edu = 10
if edu == "Masters":
    edu = 11
if edu == "Preschool":
    edu = 13

martial_stat = st.selectbox("Marital Status : ", ["divorced","married","not married"])
if martial_stat == "divorced":
    martial_stat = 0
if martial_stat == "married":
    martial_stat = 1
if martial_stat == "not married":
    martial_stat = 2

occup = st.selectbox("Occupation : ", ["Armed-Forces","Exec-managerial","Farming-fishing","Other-service","Priv-house-serv","Prof-specialty","Sales","Tech-support"])

if occup == "Armed-Forces":
    occup = 1
if occup == "Exec-managerial":
    occup = 3
if occup == "Farming-fishing":
    occup = 4
if occup == "Other-service":
    occup = 7
if occup == "Priv-house-serv":
    occup = 8
if occup == "Prof-specialty":
    occup = 9
if occup == "Sales":
    occup = 11
if occup == "Tech-support":
    occup = 12


relation = st.selectbox("Relationship : ", ["Husband","Other-relative","Not-in-family"])
if relation == "Husband":
    relation = 0
if relation == "Other-relative":
    relation = 2
if relation == "Not-in-family":
    relation = 1

race = st.selectbox("Race : ", ["Black","Other","White"])
if race == "Black":
    race = 2
if race == "Other":
    race = 3
if race == "White":
    race = 4


gender = st.selectbox("Race : ", ["Female","Male"])
if gender == "Female":
    gender = 0
if gender == "Male":
    gender = 1


c_gain = st.text_input(label = "Capital Gain : between 0-99999")

c_loss = st.text_input(label = "Capital Loss : between 0-4356")

hours_per_week = st.text_input(label = "Hours per Week : between 1-99")

native_country = st.selectbox("Native Country : ", ["Canada","France","Germany","India","United States"])

if native_country == "Canada":
    native_country = 1 
if native_country == "France":
    native_country = 9
if native_country == "Germany":
    native_country = 10 
if native_country == "India":
    native_country = 18
if native_country == "United States":
    native_country = 38

#prediction function
def Predictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    print(to_predict)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

if st.button("Predict the Salary Category"):
    with st.spinner("Predicting the result..."):
        time.sleep(1)
    
    to_predict_list = [age,w_class,edu,martial_stat,occup,relation,race,gender,int(c_gain),int(c_loss),int(hours_per_week),native_country]
    print(to_predict_list)

    result = Predictor(to_predict_list)
    print(result)        
    print("\n")
        
    if int(result)==1:
        prediction='Income more than 50K'
        st.header(prediction)
    else:
        prediction='Income less that 50K'
        st.header(prediction)
            



