#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:55:15 2019

@author: liuziwan
"""

import numpy as np
import math
import cmath
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


#预处理数据，返回每种value的一个list
def preprocessing(data, beginString, endString):
    begin = data.index(beginString)
    begin = begin + len(beginString)
    end = data.index(endString)
    value = data[begin:end]
    valueList = []
    if value == 'null':
        valueList = []
    else:
        stringList = value.split("),(")
        stringList[0] = stringList[0][1:] # only once
        stringList[-1] = stringList[-1][0:-1] # only once
        for i in range(len(stringList)):
            tmp = stringList[i].split(",")
            for j in range(len(tmp)):
                tmp[j] = float(tmp[j])
            valueList.append(tmp)
    return valueList

#补充方向数据，和加速度数据等长
def completion_orivalue(accValueList, oriValueList):
    num = len(accValueList) - len(oriValueList)
    splitLength = math.floor(len(oriValueList)/(num+1)) #向下取整
    longOriValueList = []
    index1 = 0
    index2 = 0
    for i in range(1,num+1):
        index1 = index2
        index2 = splitLength*i
        longOriValueList = longOriValueList + oriValueList[index1:index2]
        longOriValueList.append(oriValueList[index2-1])
    longOriValueList = longOriValueList + oriValueList[num*splitLength:len(oriValueList)]
    return longOriValueList
    
#将三个方向的加速度投影到地面坐标系的Z轴
def get_acc_z_gound(accValueList, longOriValueList):
    accZaxisValue = []
    for i in range(len(accValueList)):
        acc_Z = accValueList[i][0]*math.sin((longOriValueList[i][2]/180)*math.pi)+accValueList[i][1]*math.sin((longOriValueList[i][1]/180)*math.pi)+ accValueList[i][2]*cmath.sqrt(1-(math.sin((longOriValueList[i][2]/180)*math.pi))**2-(math.sin((longOriValueList[i][1]/180)*math.pi))**2)
        acc_Z = acc_Z.real
        accZaxisValue.append(acc_Z)
    return accZaxisValue

def read_data(path):
    operation = []
    root_dir2 = path
    for file in os.listdir(root_dir2):
        file_name = root_dir2 + "/" + file
        filein = open(file_name, "r")
        data = filein.read()
        operation.append(data)
        filein.close()
    alldata = operation
    return alldata

def get_features(alldata):
    features = [[] for i in range(35)]
    for k in range(len(alldata)):
        data = alldata[k]
    #获得各个list
        accValueList = preprocessing(data, "<accvalue>", "</accvalue>")
        gyroValueList = preprocessing(data, "<gyrovalue>", "</gyrovalue>")
        oriValueList = preprocessing(data, "<orivalue>", "</orivalue>")
        touchValueList = preprocessing(data, "<touchdata>", "</touchdata>")
    #构造字典，将各个value存入字典中
        behaviour = {}
        behaviour["accvalue"] = accValueList
        behaviour["gyrovalue"] = gyroValueList
        behaviour["orivalue"] = oriValueList
        behaviour["touchdata"] = touchValueList
    #获得补全后的坐标数据，并将其加入字典中
        if len(oriValueList) < len(accValueList):
            longOriValueList = completion_orivalue(accValueList, oriValueList)
        else:
            longOriValueList = oriValueList    
        behaviour['longorivalue'] = longOriValueList
    #将三个方向的加速度投影到地面坐标系的Z轴
        if len(accValueList) != 0 & len(longOriValueList) != 0:
            accZaxisValue = get_acc_z_gound(accValueList, longOriValueList)
            features[0].append(np.mean(accZaxisValue))#acc_Z_ground_mean
            features[1].append(max(accZaxisValue))#acc_Z_ground_max
            features[2].append(min(accZaxisValue))#acc_Z_ground_min
            features[3].append(np.ptp(accZaxisValue))#acc_Z_ground_range
            features[4].append(np.var(accZaxisValue))#acc_Z_ground_variance
        else:
            features[0].append(0)#acc_Z_ground_mean
            features[1].append(0)#acc_Z_ground_max
            features[2].append(0)#acc_Z_ground_min
            features[3].append(0)#acc_Z_ground_range
            features[4].append(0)#acc_Z_ground_variance
        acc_X_value = []
        acc_Y_value = []
        acc_Z_value = []
        for i in range(len(accValueList)):
            acc_X_value.append(accValueList[i][0])
            acc_Y_value.append(accValueList[i][1])
            acc_Z_value.append(accValueList[i][2])
        
        if len(acc_X_value) != 0:
            features[5].append(np.mean(acc_X_value))#acc_X_mean
            features[6].append(max(acc_X_value))#acc_X_max
            features[7].append(min(acc_X_value))#acc_X_min
            features[8].append(np.ptp(acc_X_value))#acc_X_range
            features[9].append(np.var(acc_X_value))#acc_X_variance
        else:
            features[5].append(0)#acc_X_mean
            features[6].append(0)#acc_X_max
            features[7].append(0)#acc_X_min
            features[8].append(0)#acc_X_range
            features[9].append(0)#acc_X_variance
        
        if len(acc_Y_value) != 0:
            features[10].append(np.mean(acc_Y_value))#acc_Y_mean
            features[11].append(max(acc_Y_value))#acc_Y_max
            features[12].append(min(acc_Y_value))#acc_Y_min
            features[13].append(np.ptp(acc_Y_value))#acc_Y_range
            features[14].append(np.var(acc_Y_value))#acc_Y_variance
        else:
            features[10].append(0)#acc_Y_mean
            features[11].append(0)#acc_Y_max
            features[12].append(0)#acc_Y_min
            features[13].append(0)#acc_Y_range
            features[14].append(0)#acc_Y_variance

        if len(acc_Z_value) != 0:
            features[15].append(np.mean(acc_Z_value))#acc_Z_mean
            features[16].append(max(acc_Z_value))#acc_Z_max
            features[17].append(min(acc_Z_value))#acc_Z_min
            features[18].append(np.ptp(acc_Z_value))#acc_Z_range
            features[19].append(np.var(acc_Z_value))#acc_Z_variance
        else:
            features[15].append(0)#acc_Z_mean
            features[16].append(0)#acc_Z_max
            features[17].append(0)#acc_Z_min
            features[18].append(0)#acc_Z_range
            features[19].append(0)#acc_Z_variance

        features[20].append(len(accValueList))#acc_change_time
        event = []
        press = []
        area = []
        for j in range(len(touchValueList)):
            event.append(touchValueList[j][0])
            press.append(touchValueList[j][4])
            area.append(touchValueList[j][5])
    #统计按压次数和滑动次数
        event_press_num = event.count(1.0)
        event_slide_num = event.count(2.0)
    #按压力度和面积的不同取值数
        press_num = len(set(press))
        area_num = len(set(area))
        features[21].append(event_press_num) #event_press_num
        features[22].append(event_slide_num) #event_slide_num
        features[23].append(press_num) #press_num
        features[24].append(area_num) #area_num
        gyro_X_value = []
        gyro_Y_value = []
        gyro_Z_value = []
        for t in range(len(gyroValueList)):
            gyro_X_value.append(gyroValueList[t][0])
            gyro_Y_value.append(gyroValueList[t][1])
            gyro_Z_value.append(gyroValueList[t][2])
            
        if len(gyro_X_value) != 0:
            features[25].append(np.mean(gyro_X_value))#gyro_X_mean
            features[26].append(np.ptp(gyro_X_value))#gyro_X_range
            features[27].append(np.var(gyro_X_value))#gyro_X_variance
        else:
            features[25].append(0)#gyro_X_mean
            features[26].append(0)#gyro_X_range
            features[27].append(0)#gyro_X_variance
        
        if len(gyro_Y_value) != 0:
            features[28].append(np.mean(gyro_Y_value))#gyro_Y_mean
            features[29].append(np.ptp(gyro_Y_value))#gyro_Y_range
            features[30].append(np.var(gyro_Y_value))#gyro_Y_variance
        else:
            features[28].append(0)#gyro_Y_mean
            features[29].append(0)#gyro_Y_range
            features[30].append(0)#gyro_Y_variance

        if len(gyro_Z_value) != 0:
            features[31].append(np.mean(gyro_Z_value))#gyro_Z_mean
            features[32].append(np.ptp(gyro_Z_value))#gyro_Z_range
            features[33].append(np.var(gyro_Z_value))#gyro_Z_variance
        else:
            features[31].append(0)#gyro_Z_mean
            features[32].append(0)#gyro_Z_range
            features[33].append(0)#gyro_Z_variance

        features[34].append(len(gyroValueList))#gyro_change_time
        print("完成")
        print(k)
    return features


def get_dataframe(features):
    feature_dict = {}
    feature_dict['acc_Z_ground_mean'] = features[0]
    feature_dict['acc_Z_ground_max'] = features[1]
    feature_dict['acc_Z_ground_min'] = features[2]
    feature_dict['acc_Z_ground_range'] = features[3]
    feature_dict['acc_Z_ground_variance'] = features[4]

    feature_dict['acc_X_mean'] = features[5]
    feature_dict['acc_X_max'] = features[6]
    feature_dict['acc_X_min'] = features[7]
    feature_dict['acc_X_range'] = features[8]
    feature_dict['acc_X_variance'] = features[9]

    feature_dict['acc_Y_mean'] = features[10]
    feature_dict['acc_Y_max'] = features[11]
    feature_dict['acc_Y_min'] = features[12]
    feature_dict['acc_Y_range'] = features[13]
    feature_dict['acc_Y_variance'] = features[14]

    feature_dict['acc_Z_mean'] = features[15]
    feature_dict['acc_Z_max'] = features[16]
    feature_dict['acc_Z_min'] = features[17]
    feature_dict['acc_Z_range'] = features[18]
    feature_dict['acc_Z_variance'] = features[19]

    feature_dict['acc_change_time'] = features[20]
    feature_dict['event_press_num'] = features[21]
    feature_dict['event_slide_num'] = features[22]
    feature_dict['press_num'] = features[23]
    feature_dict['area_num'] = features[24]

    feature_dict['gyro_X_mean'] = features[25]
    feature_dict['gyro_X_range'] = features[26]
    feature_dict['gyro_X_variance'] = features[27]
    feature_dict['gyro_Y_mean'] = features[28]
    feature_dict['gyro_Y_range'] = features[29]

    feature_dict['gyro_Y_variance'] = features[30]
    feature_dict['gyro_Z_mean'] = features[31]
    feature_dict['gyro_Z_range'] = features[32]
    feature_dict['gyro_Z_variance'] = features[33]
    feature_dict['gyro_change_time'] = features[34]
    df = pd.DataFrame(feature_dict)
    return df

def evaluate_data(conf_mx):
    recall = conf_mx[1][1]/(conf_mx[1][0]+conf_mx[1][1])
    precision = conf_mx[1][1]/(conf_mx[0][1]+conf_mx[1][1])
    accuracy = (conf_mx[1][1]+conf_mx[0][0])/(conf_mx[0][0]+conf_mx[0][1]+conf_mx[1][0]+conf_mx[1][1])
    F1_score = (2*recall*precision)/(recall + precision)
    print("召回率为：", recall)
    print("精确率为：", precision)
    print("正确率为：", accuracy)
    print("F1值为：", F1_score)
    

humandata = read_data("/Users/liuziwan/Downloads/operationdata/humanoperation")
machinedata = read_data("/Users/liuziwan/Downloads/operationdata/machineoperation")

human_features = get_features(humandata)
machine_features = get_features(machinedata)

df_human = get_dataframe(human_features)
df_machine = get_dataframe(machine_features)

df_human['label'] = [0]*len(df_human)
df_machine['label'] = [1]*len(df_machine)

df = pd.concat([df_machine,df_human],axis=0)
df.index = range(len(df))

#build model, randomforest
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
classifier.fit(x_train, y_train)
predict_label = classifier.predict(x_test)
conf_mx_tree = confusion_matrix(y_test, predict_label, labels=[0, 1])
print(conf_mx_tree)

evaluate_data(conf_mx_tree)




