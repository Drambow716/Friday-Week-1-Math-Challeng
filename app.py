import pandas as pd
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

#st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.markdown("<a href='#rotation-formula'>Rotation Matrix Exercise</a>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#bmi-exercise'>BMI Exercise</a>", unsafe_allow_html=True)
st.sidebar.markdown("<a href='#pca-principal-components-analysis'>PCA Exercise</a>", unsafe_allow_html=True)

st.header("Friday Week 1 Math Challenge")

#Rotation Matrix Exercise
st.header("Rotation Formula")

'''
In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean space. 
For example, using the convention below, the matrix. It rotates points in the xy-plane counterclockwise through an angle θ with respect to the x axis about the origin of a 
two-dimensional Cartesian coordinate system.
'''

st.image('./rotation.png', caption='Rotation Matrix Formula')

'''
Bellow is an example of how it's working. 
'''

thetha = 40
# Input Vector
Vector = [[3], [4]]

#  Function to calculate the trnaformed vector and Plot both.
def Rotation_Matrix(thetha, Vector):
    a = np.cos(thetha)
    b = np.sin(thetha)
    R_Thetha = ([[a, -b], [b, a]])
    Transformed_V = np.dot(R_Thetha, Vector)
    plt.plot(Vector, label="vector")
    plt.plot(Transformed_V, label="Transformed_V")
    plt.legend(loc="upper left")
    return Transformed_V, Vector

Rotation_Matrix(thetha, Vector)

st.pyplot()

#BMI Part
st.header("BMI Exercise")
'''In this exercise we have collected data about 500 people, and the goal is to study each group in details,
give a general overview. But also going in detail and exploring other intersting fact of weight by gender.'''

df = pd.read_csv("https://raw.githubusercontent.com/chriswmann/datasets/master/500_Person_Gender_Height_Weight_Index.csv")

st.subheader("Here is the data ")
st.dataframe(df)

mean_vector = [df.Weight.mean(), df.Height.mean()]

mean_for_weight = 0
for i in df.Weight:
    mean_for_weight += i

mean_for_Height = 0
for j in df.Height:
    mean_for_Height += j

a = df.Weight
b = df.Height
c = df.Weight.mean()
d = df.Height.mean()
'''
Ploting the data  together with the mean that is visible in the middle in another Color.

'''
plt.scatter(a, b)
plt.scatter(c, d)
st.pyplot()

# dividing the mean data by indexes
index_1 = df.loc[df["Index"] == 1]
index_2 = df.loc[df["Index"] == 2]
index_3 = df.loc[df["Index"] == 3]
index_4 = df.loc[df["Index"] == 4]
index_5 = df.loc[df["Index"] == 5]
#df_male = df.loc[df["Gender"] == "Male"]
#print(index_2)

#medium weight for index 1
w1 = 0
for i in index_1.Weight:
    w1 += i
sum = len(index_1)
mean_w1 = w1 / sum


#medium height for index 1
h1 = 0
for i in index_1.Height:
    h1 += i
sum = len(index_1)
mean_h1 = h1 / sum

w2 = 0
for i in index_2.Weight:
    w2 += i
sum = len(index_2)
mean_w2 = w2 / sum

h2 = 0
for i in index_2.Height:
    h2 += i
sum = len(index_2)
mean_h2 = h2 / sum

w3 = 0
for i in index_3.Weight:
    w3 += i
sum = len(index_3)
mean_w3 = w3 / sum

h3 = 0
for i in index_3.Height:
    h3 += i
sum = len(index_3)
mean_h3 = h3 / sum

w4 = 0
for i in index_4.Weight:
    w4 += i
sum = len(index_4)
mean_w4 = w4 / sum

h4 = 0
for i in index_4.Height:
    h4 += i
sum = len(index_4)
mean_h4 = h4 / sum

w5 = 0
for i in index_5.Weight:
    w5 += i
sum = len(index_5)
mean_w5 = w5 / sum

#medium height for index 5
h5 = 0
for i in index_5.Height:
    h5 += i
sum = len(index_5)
mean_h5 = h5 / sum

a = df.Weight
b = df.Height
c = df.Weight.mean()
d = df.Height.mean()
plt.scatter(a, b)
plt.scatter(c, d, color="red")
plt.scatter(mean_w1, mean_h1, color="green")
plt.scatter(mean_w2, mean_h2, color='black')
plt.scatter(mean_w3, mean_h3, color='yellow')
plt.scatter(mean_w4, mean_h4, color='brown')
plt.scatter(mean_w5, mean_h5, color='orange')

''' 

Ploting the mean of the all points as above together with the means of the different groups, all with different colors.
 
'''
st.pyplot()

#PCA Part
st.header("PCA : Principal components Analysis")

''' PCA is a most widely used tool in exploratory data analysis and in machine learning for predictive models. Moreover, 
PCA is an unsupervised statistical technique used to examine the interrelations among a set of variables. It is also 
known as a general factor analysis where regression determines a line of best fit.'''

df = pd.read_csv("https://raw.githubusercontent.com/chriswmann/datasets/master/500_Person_Gender_Height_Weight_Index.csv")
df_new = df.replace('Female', 1).replace('Male', 0)
pca = PCA(n_components=2)
proj = pca.fit_transform(df_new)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.scatter(proj[:, 0], proj[:, 1], c=df_new.Index/5, cmap="Paired")
ax1.set_title('BMI')

ax2.scatter(proj[:, 0], proj[:, 1], c=df_new.Gender, cmap="Paired")
ax2.set_title('Gender')

st.pyplot()



