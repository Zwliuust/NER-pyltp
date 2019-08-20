#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:41:16 2019

@author: liuziwan
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_data(path):
    operation = {}
    root_dir2 = path
    for file in os.listdir(root_dir2):
        file_name = root_dir2 + "/" + file
        filein = open(file_name, "r")
        data = filein.read()
        operation[file] = data
        filein.close()
    alldata = operation
    return alldata

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

def get_machine_touch_features(alldata):
    features = [[] for i in range(26)]
    for key in alldata:
        data = alldata[key]
        features[25].append(key)
    #获得各个list
        touchValueList = preprocessing(data, "<touchdata>", "</touchdata>")
    #构造字典，将各个value存入字典中
        features[0].append(len(touchValueList)) #the total amount of event
        tmp = []
        x_changetime_list = []
        y_changetime_list = []
        press_changetime_list = []
        area_changetime_list = []
        x_axis = []
        y_axis = []
        press = []
        area = []
        x_axis_mean = []
        y_axis_mean = []
        press_mean = []
        area_mean = []
        for i in range(len(touchValueList)):
            tmp.append(touchValueList[i])
            if touchValueList[i][0] == 3.0:
                for j in range(len(tmp)):
                    x_axis.append(tmp[j][2])
                    y_axis.append(tmp[j][3])
                    press.append(tmp[j][4])
                    area.append(tmp[j][5])
                x_axis_mean.append(np.mean(x_axis))
                y_axis_mean.append(np.mean(y_axis))
                press_mean.append(np.mean(press))
                area_mean.append(np.mean(area))
                x_diff_num = len(set(x_axis))
                x_changetime_list.append(x_diff_num)
                y_diff_num = len(set(y_axis))
                y_changetime_list.append(y_diff_num)
                press_diff_num = len(set(press))
                press_changetime_list.append(press_diff_num)
                area_diff_num = len(set(area))
                area_changetime_list.append(area_diff_num)
                tmp = []
        if len(x_axis_mean)>0:
            x_axis_mean = x_axis_mean[0]
        else:
            x_axis_mean = 0
            
        if len(y_axis_mean)>0:
            y_axis_mean = y_axis_mean[0]
        else:
            y_axis_mean = 0
        
        if len(press_mean)>0:
            press_mean = press_mean[0]
        else:
            press_mean = 0
            
        if len(area_mean)>0:
            area_mean = area_mean[0]
        else:
            area_mean = 0
            
        if len(x_changetime_list)>0:
            x_changetime_mean = np.mean(x_changetime_list)#mean of the x axis changes in each event groupmean
            x_changetime_var = np.var(x_changetime_list) #variance of x axis change times
        else:
            x_changetime_mean = 0
            x_changetime_var = 0
            
        if len(y_changetime_list)>0:
            y_changetime_mean = np.mean(y_changetime_list)#mean of the y axis changes in each event groupmean
            y_changetime_var = np.var(y_changetime_list) #variance of y axis change times
        else:
            y_changetime_mean = 0
            y_changetime_var = 0
            
        if len(press_changetime_list)>0:
            press_changetime_mean = np.mean(press_changetime_list)#mean of the press changes in each event groupmean
            press_changetime_var = np.var(press_changetime_list) #variance of press change times
        else:
            press_changetime_mean = 0
            press_changetime_var = 0
            
        if len(area_changetime_list)>0:
            area_changetime_mean = np.mean(area_changetime_list)#mean of the area changes in each event groupmean
            area_changetime_var = np.var(area_changetime_list) #variance of area change times
        else:
            area_changetime_mean = 0
            area_changetime_var = 0
        global_x_axis = []
        global_y_axis = []
        global_press = []
        global_area = []
        for i in range(len(touchValueList)):
            global_x_axis.append(touchValueList[i][2])
            global_y_axis.append(touchValueList[i][3])
            global_press.append(touchValueList[i][4])
            global_area.append(touchValueList[i][5])
            
        if len(global_x_axis) >0:
            global_x_mean = np.mean(global_x_axis)
            global_x_var = np.var(global_x_axis)
            global_x_range = np.ptp(global_x_axis)
        else:
            global_x_mean = 0
            global_x_var = 0
            global_x_range = 0
            
        if len(global_y_axis) >0:
            global_y_mean = np.mean(global_y_axis)
            global_y_var = np.var(global_y_axis)
            global_y_range = np.ptp(global_y_axis)
        else:
            global_y_mean = 0
            global_y_var = 0
            global_y_range = 0
        
        if len(global_press)>0:
            global_press_mean = np.mean(global_press)
            global_press_var = np.var(global_press)
            global_press_range = np.ptp(global_press)
        else:
            global_press_mean = 0
            global_press_var = 0
            global_press_range = 0
        
        if len(global_area)>0:
            global_area_mean = np.mean(global_area)
            global_area_var = np.var(global_area)
            global_area_range = np.ptp(global_area)
        else:
            global_area_mean = 0
            global_area_var = 0
            global_area_range = 0
            
        features[1].append(x_changetime_mean)
        features[2].append(x_changetime_var)
        features[3].append(y_changetime_mean)
        features[4].append(y_changetime_var)
        features[5].append(press_changetime_mean)
        features[6].append(press_changetime_var)
        features[7].append(area_changetime_mean)
        features[8].append(area_changetime_var)
        features[9].append(global_x_mean)
        features[10].append(global_x_var)
        features[11].append(global_x_range)
        features[12].append(global_y_mean)
        features[13].append(global_y_var)
        features[14].append(global_y_range)
        features[15].append(global_press_mean)
        features[16].append(global_press_var)
        features[17].append(global_press_range)
        features[18].append(global_area_mean)
        features[19].append(global_area_var)
        features[20].append(global_area_range)
        features[21].append(x_axis_mean) #在第一组事件中，x轴坐标的均值
        features[22].append(y_axis_mean) #在第一组事件中，y轴坐标的均值
        features[23].append(press_mean)  #在第一组事件中，压力的均值
        features[24].append(area_mean)   #在第一组事件中，按压面积的均值
    return features
        
        
def machine_get_dataframe(features):
    feature_dict = {}
    feature_dict['total_events'] = features[0]
    feature_dict['x_changetime_mean'] = features[1]
    feature_dict['x_changetime_var'] = features[2]
    feature_dict['y_changetime_mean'] = features[3]
    feature_dict['y_changetime_var'] = features[4]

    feature_dict['press_changetime_mean'] = features[5]
    feature_dict['press_changetime_var'] = features[6]
    feature_dict['area_changetime_mean'] = features[7]
    feature_dict['area_changetime_var'] = features[8]
    feature_dict['global_x_mean'] = features[9]

    feature_dict['global_x_var'] = features[10]
    feature_dict['global_x_range'] = features[11]
    feature_dict['global_y_mean'] = features[12]
    feature_dict['global_y_var'] = features[13]
    feature_dict['global_y_range'] = features[14]

    feature_dict['global_press_mean'] = features[15]
    feature_dict['global_press_var'] = features[16]
    feature_dict['global_press_range'] = features[17]
    feature_dict['global_area_mean'] = features[18]
    feature_dict['global_area_var'] = features[19]

    feature_dict['global_area_range'] = features[20]
    feature_dict['x_axis_mean'] = features[21]
    feature_dict['y_axis_mean'] = features[22]
    feature_dict['press_mean'] = features[23]
    feature_dict['area_mean'] = features[24]
    feature_dict['file_name'] = features[25]
    
    df = pd.DataFrame(feature_dict)
    return df

machinedata = read_data("/Users/liuziwan/Downloads/operationdata/machineoperation")
machine_features = get_machine_touch_features(machinedata)
machine_df = machine_get_dataframe(machine_features)
df_array = machine_df.values[:,:-1]
#聚类类别
n_clusters = 7
#KMeans聚类
cls = KMeans(n_clusters).fit(df_array)
#给出每个样本的所属类别
pre_class = cls.labels_
file_name = list(machine_df.iloc[:,-1])

#输出文件名及对应的预测类别
result_dict = {}
result_dict['file'] = file_name
result_dict['label'] = pre_class


#画图, 倒三角、正方形、五角星、加号、钻石、x号、圆圈
markers = ['v','s','*','+','D', 'x', 'o']
for i in range(n_clusters):
    members = cls.labels_ == i#members是布尔数组
    plt.scatter(df_array[members,0],df_array[members,1],s = 20,marker = markers[i],c = 'r',alpha=0.5)#画与menbers数组中匹配的点

plt.title('machine_classes')
plt.show()

machinedata
