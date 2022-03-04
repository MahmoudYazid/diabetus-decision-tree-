from sklearn import tree
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\ahmed\\Desktop\\pyprojects\\diabetes.csv")


y = np.array(data['Outcome'])

x = pd.DataFrame(data[['Pregnancies', 'Glucose', 'BloodPressure',
                       'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]).to_numpy()


x=np.array(x)

collection_arr = [[1, 93, 70, 31, 0, 30.4, 0.315, 23, 0],[6, 190, 92, 0, 0, 35.5, 0.278, 66, 1],[2, 88, 58, 26, 16, 28.4, 0.766, 22, 0]
    ,[9, 170, 74, 31, 0, 44, 0.403, 43, 1]
    ,[9, 89, 62, 0, 0, 22.5, 0.142, 33, 0]
    ,[10, 101, 76, 48, 180, 32.9, 0.171, 63, 0]
    ,[2, 122, 70, 27, 0, 36.8, 0.34, 27, 0]
    ,[5, 121, 72, 23, 112, 26.2, 0.245, 30, 0]
,[1	,126	,60	,0	,0	,30.1	,0.349	,47	,1]]



model= tree.DecisionTreeClassifier()
model.fit(x,y)
result_test = [0, 1, 0, 1, 0, 0, 0, 0, 1]
no_of_result=0
n=0
for arr in collection_arr:
    pred_arr = arr
    newsize_to_predict=np.resize(pred_arr, [1,8])


    newprd=model.predict(newsize_to_predict)
    print(newprd, " ", "expected", " ", ":", " ", result_test[n])
    print()
    n+=1



