# -*- coding: utf-8 -*-
# Copyright 2023 Tencent Inc.  All rights reserved.
# Author: broccozhang@tencent.com


import time
import math
from scipy.stats import norm
import json
import sys
import os

import pandas as pd
import numpy as np
import seaborn as sns
import time
from graphviz import Digraph
import textwrap
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

from .. import create_sql_instance, clickhouse_create_view,clickhouse_drop_view


# global sql_instance
# sql_instance = ais.create()

# parse params
def SelectSchema(schema):
    return schema


def FeatNames(schema):
    if ',' in schema:
        schemaArray = schema.split(",")
    else:
        schemaArray = [schema]
    return schemaArray


def FilterSchema(schemaArray):
    tmp = [i + " is not null" for i in schemaArray]
    return " and ".join(tmp)


# CausalTree class
class CausalTreeclass():
    def __init__(
            self,
            dfName='dat',
            threshold = 0.01,
            maxDepth=3,
            whereCond="",
            nodePosition="L",
            impurity=0,
            father_split_feature="",
            father_split_feature_Categories=[],
            whereCond_new=[],
            nodesSet=[]
    ):

        self.dfName = dfName
        self.threshold = threshold
        self.maxDepth = maxDepth
        self.whereCond = whereCond
        self.nodePosition = nodePosition
        self.impurity = impurity
        self.father_split_feature = father_split_feature
        self.father_split_feature_Categories = father_split_feature_Categories
        self.whereCond_new = whereCond_new
        self.nodesSet = nodesSet

        self.leftNode = 0
        self.rightNode = 0
        self.nodeSize = 0
        self.controlcount = 0
        self.treatcount = 0
        self.splitFeat = ""
        self.splitIndex = 0
        self.splitpoint = ""
        self.prediction = 0
        self.maxImpurGain = 0
        self.isLeaforNot = False
        self.splitpoint_pdf = pd.DataFrame()
        self.allsplitpoint_pdf = pd.DataFrame()
        self.featvalues_dict = dict()
        self.inferenceSet = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.dat_cnt = 0
        self.depth = 3
        self.sql_instance = create_sql_instance()

    def get_global_values(self, dat_cnt, depth, featNames):
        self.dat_cnt = dat_cnt
        self.depth = depth
        self.featNames = featNames

    def compute_df(self):
        threshold = self.threshold
        

        if (self.maxDepth == 0):
            self.isLeaforNot = True
            #             print("Reach the maxDepth, stop as a leaf node")
            return
        if (self.whereCond != ""):
            whereCond_ = "AND" + self.whereCond
        else:
            whereCond_ = ""

        dfName = self.dfName

        sql_instance = self.sql_instance
        row1 = sql_instance.sql(f'''SELECT count(Y) as cnt,avg(Y) as mean,varSamp(Y) as var FROM {dfName} 
                                WHERE treatment = 1 and if_test = 0 {whereCond_}'''.format(whereCond_=whereCond_,
                                                                                           dfName=dfName))
        row1 = pd.DataFrame(list(row1), columns=['cnt', 'mean', 'var'])

        row0 = sql_instance.sql(f'''SELECT count(Y) as cnt,avg(Y) as mean,varSamp(Y) as var FROM {dfName} 
                                WHERE treatment = 0 and if_test = 0 {whereCond_}'''.format(whereCond_=whereCond_,
                                                                                           dfName=dfName))
        row0 = pd.DataFrame(list(row0), columns=['cnt', 'mean', 'var'])
        
        
        self.treatcount = row1['cnt'][0]
        self.controlcount = row0['cnt'][0]
        self.nodeSize = self.controlcount + self.treatcount
#         print(self.getTreeID(),self.nodeSize)

        if (self.nodeSize / self.dat_cnt < threshold):
            self.isLeaforNot = True
            #             print("sample size is too small, stop as a leaf node")
            return
        if self.nodeSize == 0:
            self.isLeaforNot = True
            #             print(" sample size = 0,  stop as a leaf node")
            return
        if row1['var'][0] == 0 or row0['var'][0] == 0:
            self.isLeaforNot = True
            #             print(" var = 0, stop as a leaf node ")
            return

        t1 = time.time()
        allsplitpoint_data = []

        for featName in self.featNames:
            sql = f''' SELECT '{featName}' as featName,{featName} as featValue,\
                        sum(if(treatment=1,Y,0)) AS y1, \
                        sum(if(treatment=1,Y*Y,0)) AS y1_square, \
                        sum(if(treatment=1,1,0)) as cnt1, \
                        sum(if(treatment=0,Y,0)) AS y0, \
                        sum(if(treatment=0,Y*Y,0)) AS y0_square, \
                        sum(if(treatment=0,1,0)) as cnt0 \
                  FROM {dfName} 
                  where if_test = 0 {whereCond_}\
                  GROUP BY '{featName}',{featName} '''
            featureTau = sql_instance.sql(sql)
            # print(featName,featureTau)
            data = []
            for i in featureTau:
                allsplitpoint_data.append(list(i))
                data.append(list(i))
                # print(allsplitpoint_data)
            try:
                pd.DataFrame(data,
                                           columns=['featName', 'featValue', 'y1', 'y1_square', 'cnt1', 'y0',
                                                    'y0_square', 'cnt0'])
            except:
                print(featName,i,data)
        splitpoint_data_pdf = pd.DataFrame(allsplitpoint_data,
                                           columns=['featName', 'featValue', 'y1', 'y1_square', 'cnt1', 'y0',
                                                    'y0_square', 'cnt0'])

        splitpoint_data_pdf['tau'] = splitpoint_data_pdf['y1'] / splitpoint_data_pdf['cnt1'] - splitpoint_data_pdf[
            'y0'] / splitpoint_data_pdf['cnt0']
        splitpoint_data_pdf = splitpoint_data_pdf.sort_values(by=['featName', 'tau'], ascending=False)
        tmp = splitpoint_data_pdf.groupby(by=['featName'], as_index=False)['featValue'].count()
        tmp.columns = ['featName', 'featpoint_num']
        splitpoint_data_pdf = pd.merge(splitpoint_data_pdf, tmp, on=['featName'])
        splitpoint_data_pdf = splitpoint_data_pdf.dropna()
        splitpoint_data_pdf_copy = splitpoint_data_pdf.copy()
        self.allsplitpoint_pdf = splitpoint_data_pdf_copy

        splitpoint_data_pdf = splitpoint_data_pdf[splitpoint_data_pdf.featpoint_num > 1]
        featNames_ = list(set(splitpoint_data_pdf['featName']))
        featValuesall_list = []
        featName_list = []
        featValue_list = []
        splitpoint_list = []
        cnt0_list = []
        cnt1_list = []
        splitpoint_list = []
        for featName_ in featNames_:
            featValuesall = list(splitpoint_data_pdf[(splitpoint_data_pdf['featName'] == featName_)]['featValue'])
            cnt0s = list(splitpoint_data_pdf[(splitpoint_data_pdf['featName'] == featName_)]['cnt0'])
            cnt1s = list(splitpoint_data_pdf[(splitpoint_data_pdf['featName'] == featName_)]['cnt1'])
            featValuesall_list.append(featValuesall)

            for i in range(len(featValuesall) - 1):
                cnt0 = np.sum(cnt0s[0:i + 1])
                cnt1 = np.sum(cnt1s[0:i + 1])
                cnt0_list.append(cnt0)
                cnt1_list.append(cnt1)
                featName_list.append(featName_)
                splitpoint_list.append(dict({featName_: featValuesall[0:i + 1]}))
                featValue_list.append(featValuesall[0:i + 1])

        splitpoint_pdf = pd.DataFrame(zip(featName_list, featValue_list, splitpoint_list, cnt0_list, cnt1_list),
                                      columns=['featName', 'featValue', 'splitpoint', 'cnt0', 'cnt1'])
        #         print("splitpoint_pdf_before:\n",splitpoint_pdf)
        splitpoint_pdf = splitpoint_pdf[(splitpoint_pdf['cnt0'] > 100) & (splitpoint_pdf['cnt1'] > 100) &
                                        (splitpoint_pdf['cnt0'] + splitpoint_pdf['cnt1'] > threshold * self.dat_cnt) &
                                        (self.nodeSize - splitpoint_pdf['cnt0'] - splitpoint_pdf[
                                            'cnt1'] > threshold * self.dat_cnt)]
        self.splitpoint_pdf = splitpoint_pdf
        self.featvalues_dict = dict(zip(featNames_, featValuesall_list))

        #         print("splitpoint_pdf:\n",self.splitpoint_pdf[['featName','featValue']])
        t2 = time.time()

    #         print("compute allsplitpoint_data time: ",t2-t1)

    def splitcond_sql(self,splitpoint):
        
        featName = list(splitpoint.keys())[0]
        featValue = list(splitpoint.values())[0]

        for i in self.sql_instance.sql(f"desc {self.dfName}"):
            if featName == i[0]:
                featType = i[1]
        if featType == 'String':
            featValue = ','.join(["'"+str(i)+"'" for i in featValue])
        else:
            featValue = ','.join([str(i) for i in featValue])

        left_condition_tree = f'''({featName} in ({featValue}))'''
        right_condition_tree = f'not ({featName} in ({featValue}))'''
        return left_condition_tree, right_condition_tree

    def calculate_impurity_new(self, x):
        allsplitpoint_pdf = self.allsplitpoint_pdf  # TODO
        featName = x['featName']
        featValue = x['featValue']

        # left impurity
        (y1, y1_square, cnt1, y0, y0_square, cnt0) = list(
            allsplitpoint_pdf[(allsplitpoint_pdf['featValue'].isin(featValue)) & (
                    allsplitpoint_pdf['featName'] == featName)][
                ['y1', 'y1_square', 'cnt1', 'y0', 'y0_square', 'cnt0']].sum())
        y1 = y1 / cnt1
        y0 = y0 / cnt0
        y1_square = y1_square / cnt1
        y0_square = y0_square / cnt0
        tau = y1 - y0
        tr_var = y1_square - y1 ** 2
        con_var = y0_square - y0 ** 2
        left_effect = 0.5 * tau * tau * (cnt1 + cnt0) - 0.5 * 2 * (cnt1 + cnt0) * (tr_var / cnt1 + con_var / cnt0)

        # right impurity
        (y1, y1_square, cnt1, y0, y0_square, cnt0) = list(
            allsplitpoint_pdf[(~(allsplitpoint_pdf['featValue'].isin(featValue))) & (
                    allsplitpoint_pdf['featName'] == featName)][
                ['y1', 'y1_square', 'cnt1', 'y0', 'y0_square', 'cnt0']].sum())
        y1 = y1 / cnt1
        y0 = y0 / cnt0
        y1_square = y1_square / cnt1
        y0_square = y0_square / cnt0

        tau = y1 - y0
        tr_var = y1_square - y1 ** 2
        con_var = y0_square - y0 ** 2
        right_effect = 0.5 * tau * tau * (cnt1 + cnt0) - 0.5 * 2 * (cnt1 + cnt0) * (tr_var / cnt1 + con_var / cnt0)
        return left_effect, right_effect

    def calculate_impurity_original(self):
        sql = f''' select\
                        sum(if(treatment=1,1,0)) as cnt1,  \
                        sum(if(treatment=1,Y,0))/sum(if(treatment=1,1,0)) as y1, \
                        sum(if(treatment=1,Y*Y,0))/sum(if(treatment=1,1,0)) as y1_square, \
                        sum(if(treatment=0,1,0)) as cnt0,  \
                        sum(if(treatment=0,Y,0))/sum(if(treatment=0,1,0)) as y0, \
                        sum(if(treatment=0,Y*Y,0))/sum(if(treatment=0,1,0)) as y0_square \
                 from {self.dfName} 
                 where if_test = 0 '''
        sql_instance = self.sql_instance
        ImpurityData_df = sql_instance.sql(sql)
        row = pd.DataFrame(ImpurityData_df, columns=['cnt1', 'y1', 'y1_square', 'cnt0', 'y0', 'y0_square']).iloc[0, :]
        #         print(row)
        cnt1, y1, y1_square, cnt0, y0, y0_square = row.cnt1, row.y1, row.y1_square, row.cnt0, row.y0, row.y0_square
        tau = y1 - y0
        tr_var = y1_square - y1 ** 2
        con_var = y0_square - y0 ** 2
        effect = 0.5 * tau * tau * (cnt1 + cnt0) - 0.5 * 2 * (cnt1 + cnt0) * (tr_var / cnt1 + con_var / cnt0)
        return effect

    def getTreeID(self):

        nodePosition = self.nodePosition
        lengthStr = len(nodePosition)
        if lengthStr == 1:
            result = 1
        else:
            result = sum([math.pow(2, i) for i in range(lengthStr - 1)]) + 1
        for i in range(lengthStr):
            x = 1 if nodePosition[i] == 'R' else 0
            result += math.pow(2, lengthStr - i - 1) * x
        return result

    def get_node_type(self):

        if (self.isLeaforNot == False):
            return "internal"
        else:
            return "leaf"

    def getBestSplit(self):

        #         print("-----getBestSplit-----")
        splitpoint_pdf = self.splitpoint_pdf
        if (splitpoint_pdf.count()[0] == 0):
            self.isLeaforNot = True
            #             print("no split points that satisfy the condition,stop splitting as a leaf node")
            return

        t1 = time.time()
        splitpoint_pdf[['leftImpurity', 'rightImpurity']] = list(
            splitpoint_pdf.apply(self.calculate_impurity_new, axis=1))
        splitpoint_pdf['ImpurityGain'] = splitpoint_pdf['leftImpurity'] + splitpoint_pdf[
            'rightImpurity'] - self.impurity
        splitpoint_pdf = splitpoint_pdf.sort_values(by="ImpurityGain", ascending=False)
        t2 = time.time()
        #         print("get best split time ",t2-t1)

        #         print('all splitpoints \n',splitpoint_pdf[['splitpoint','ImpurityGain']])

        splitpoint_pdf = splitpoint_pdf[(splitpoint_pdf['ImpurityGain'] > 0)]
        
        self.splitpoint_pdf = splitpoint_pdf[['featName','featValue','splitpoint','ImpurityGain']]
        
        if (splitpoint_pdf.count()[0] == 0):
            self.isLeaforNot = True
            #             print("no split points that satisfy the condition,stop splitting as a leaf node")
            return

        bestFeatureName = splitpoint_pdf.iloc[0]["featName"]
        bestFeatureIndex = splitpoint_pdf.iloc[0]["featValue"]
        bestleftImpurity = splitpoint_pdf.iloc[0]["leftImpurity"]
        bestrightImpurity = splitpoint_pdf.iloc[0]["rightImpurity"]
        maxImpurityGain = splitpoint_pdf.iloc[0]["ImpurityGain"]
        bestsplitpoint = splitpoint_pdf.iloc[0]["splitpoint"]
        self.maxImpurGain = maxImpurityGain
        self.splitFeat = bestFeatureName
        self.splitIndex = bestFeatureIndex
        self.splitpoint = bestsplitpoint
        self.leftImpurity = bestleftImpurity
        self.rightImpurity = bestrightImpurity

    def buildTree(self):
        #         global nodesSet
        self.nodesSet.append(self)
        #         print(self.nodesSet)
        if self.whereCond == '':
            self.impurity = self.calculate_impurity_original()
        #             print("root impurity:",self.impurity)

        self.compute_df()
        if (self.isLeaforNot):
            return
        else:
            self.getBestSplit()
            if (self.isLeaforNot):
                return
        #         print("end split")
        whereConditionSql = "" if (self.whereCond == "") else self.whereCond + " AND "
        leftcondition_tree, rightcondition_tree = self.splitcond_sql(self.splitpoint)
        leftCond = f"{whereConditionSql} ( {leftcondition_tree} ) "
        rightCond = f"{whereConditionSql} ( {rightcondition_tree} ) "
        leftChildDfName = self.nodePosition + "L"
        rightChildDfName = self.nodePosition + "R"


        # TODO：
        father_split_feature = self.splitFeat
        Categories = self.featvalues_dict[father_split_feature]
        left_Categories = self.splitpoint[father_split_feature]
        right_Categories = []
        for i in Categories:
            if i not in left_Categories:
                right_Categories.append(i)
        left_tmp = {}
        right_tmp = {}
        #         print("self.whereCond_new",self.whereCond_new)
        #         print(father_split_feature,left_Categories,right_Categories)
        left_tmp[father_split_feature] = left_Categories
        right_tmp[father_split_feature] = right_Categories
        leftCond_new = self.whereCond_new.copy()
        leftCond_new.append(left_tmp)
        rightCond_new = self.whereCond_new.copy()
        rightCond_new.append(right_tmp)
        #         print("leftCond_new,rightCond_new",leftCond_new,rightCond_new)

        leftChild = CausalTreeclass(self.dfName,self.threshold, self.maxDepth - 1, leftCond, self.nodePosition + "L",
                                    self.leftImpurity, father_split_feature, left_Categories, leftCond_new,
                                    self.nodesSet)
        rightChild = CausalTreeclass(self.dfName,self.threshold, self.maxDepth - 1, rightCond, self.nodePosition + "R",
                                     self.rightImpurity, father_split_feature, right_Categories, rightCond_new,
                                     self.nodesSet)
        leftChild.get_global_values(self.dat_cnt, self.depth, self.featNames)
        rightChild.get_global_values(self.dat_cnt, self.depth, self.featNames)

        self.leftNode = leftChild
        self.rightNode = rightChild
        #         print("-----split result----")
        #         print("maxImpurGain:",self.maxImpurGain)
        #         print("splitpoint:",self.splitpoint)
        #         print("leftsplitcond,rightsplitcond:",leftcondition_tree,rightcondition_tree)
        #         print("leftImpurity,rightImpurity:",self.leftImpurity,self.rightImpurity)
        #         print("-----------------------start leftNode - buildTree -- maxDepth: {maxDepth}, nodePosition: {nodePosition}-----------------------".format(maxDepth=self.maxDepth-1,nodePosition=self.nodePosition + "L"))
        self.leftNode.buildTree()
        #         print("-----------------------start rightNode - buildTree -- maxDepth: {maxDepth}, nodePosition: {nodePosition}-----------------------".format(maxDepth=self.maxDepth-1,nodePosition=self.nodePosition + "R"))
        self.rightNode.buildTree()

    def visualization(self):
        if (self.isLeaforNot):
            return "\n" + "Prediction is: " + str(self.prediction) + " " + self.whereCond
        else:
            return "Level" + str(self.maxDepth) + "  " + self.splitpoint + self.maxImpurGain + "\n" + self.leftNode.visualization() + "\n" + self.rightNode.visualization()

    def ComputePvalueAndCI(self, zValue, prediction, meanStd):

        pvalue = 2 * (1 - norm.cdf(x=abs(zValue), loc=0, scale=1))
        lowerCI = prediction - 1.96 * (meanStd)
        upperCI = prediction + 1.96 * (meanStd)
        return (pvalue, lowerCI, upperCI)

    def predictNode(self, ate=0, cnt=1):

        if (self.whereCond != ""):
            whereCond_ = "AND" + self.whereCond
        else:
            whereCond_ = ""

        ate = ate
        sql = f'''SELECT \
        treatment, count(*) as cnt, avg(Y) as mean, varSamp(Y) as var \
        FROM {self.dfName} \
        WHERE treatment in (0,1) {whereCond_} and if_test=1 \
        GROUP BY treatment \
        '''
        sql_instance = self.sql_instance
        statData = sql_instance.sql(sql)
#         if statData=='success':
        try:
            statData_pdf = pd.DataFrame(statData, columns=['treatment', 'cnt', 'mean', 'var'])
            y1_column = statData_pdf[statData_pdf['treatment'] == 1]
            y0_column = statData_pdf[statData_pdf['treatment'] == 0]
            treatedCount = int(list(y1_column['cnt'])[0])
            controlCount = int(list(y0_column['cnt'])[0])
            treatedLabelAvg = float(list(y1_column['mean'])[0])
            controlLabelAvg = float(list(y0_column['mean'])[0])
            treatedLabelVar = float(list(y1_column['var'])[0])
            controlLabelVar = float(list(y0_column['var'])[0])
            self.prediction = treatedLabelAvg - controlLabelAvg
        except:
            print(statData,sql)

       

        level = self.depth - self.maxDepth
        TreeID = self.getTreeID()
        isLeaf = True if (self.isLeaforNot) else False
        whereCond_new = self.whereCond_new
        ratio = (treatedCount + controlCount) / cnt

        if treatedLabelAvg == 0 or controlLabelAvg == 0:
            self.inferenceSet = list(
                [TreeID, level, isLeaf, whereCond_, whereCond_new, ratio, self.prediction, treatedCount, controlCount,
                 treatedLabelAvg, controlLabelAvg, math.sqrt(treatedLabelVar), math.sqrt(controlLabelVar),
                 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 0
                 ])  # TODO:fix zero bug
        else:
            estPoint1 = treatedLabelAvg - controlLabelAvg
            std1 = math.sqrt(treatedLabelVar / treatedCount + controlLabelVar / controlCount)
            zValue1 = estPoint1 / std1
            (pvalue1, lowerCI1, upperCI1) = self.ComputePvalueAndCI(zValue1, estPoint1, std1)

            estPoint2 = treatedLabelAvg - controlLabelAvg - ate
            std2 = std1
            zValue2 = estPoint2 / std2
            (pvalue2, lowerCI2, upperCI2) = self.ComputePvalueAndCI(zValue2, estPoint2, std2)

            estPoint3 = treatedLabelAvg / controlLabelAvg - 1
            std3 = std1
            zValue3 = zValue1
            pvalue3 = pvalue1
            lowerCI3 = estPoint3 - 1.96 * (std1) * estPoint3 / estPoint1
            upperCI3 = estPoint3 + 1.96 * (std1) * estPoint3 / estPoint1

            estPoint4 = (treatedLabelAvg - ate) / controlLabelAvg - 1
            std4 = std2
            zValue4 = zValue2
            pvalue4 = pvalue2

            if TreeID == 1:
                estPoint2, estPoint4 = 0, 0
                lowerCI4 = 0
                upperCI4 = 0
            else:
                lowerCI4 = estPoint4 - 1.96 * (std2) * estPoint4 / estPoint2
                upperCI4 = estPoint4 + 1.96 * (std2) * estPoint4 / estPoint2


            self.inferenceSet = list(
                [TreeID, level, isLeaf, whereCond_, whereCond_new, ratio, self.prediction, treatedCount, controlCount,
                 treatedLabelAvg, controlLabelAvg, math.sqrt(treatedLabelVar), math.sqrt(controlLabelVar),
                 estPoint1, std1, pvalue1, zValue1, lowerCI1, upperCI1,
                 estPoint2, std2, pvalue2, zValue2, lowerCI2, upperCI2,
                 estPoint3, std3, pvalue3, zValue3, lowerCI3, upperCI3,
                 estPoint4, std4, pvalue4, zValue4, lowerCI4, upperCI4
                 ])


# define tree structure
class CTDecisionNode:

    def __init__(
            self,
            node_id=0,
            nodeType="",
            node_order="",
            nodePosition="",
            count_ratio=0,
            controlCount=0,
            treatedCount=0,
            treatedAvg=0,
            controlAvg=0,
            splitType="",
            gain=0,
            impurity=0,
            prediction=0,
            prediction_new=[],
            featureName="",
            featureIndex="",
            father_split_feature="",
            father_split_feature_Categories=[],
            pvalues=[],
            qvalues=[],
            children=None,
            whereCond="",
            whereCond_new=[]

    ):
        self.node_id = node_id
        self.nodeType = nodeType
        self.node_order = node_order
        self.nodePosition = nodePosition
        self.count_ratio = count_ratio
        self.controlCount = controlCount
        self.treatedCount = treatedCount
        self.treatedAvg = treatedAvg
        self.controlAvg = controlAvg
        self.splitType = splitType
        self.gain = gain
        self.impurity = impurity
        self.prediction = prediction
        self.prediction_new = prediction_new
        self.featureName = featureName
        self.featureIndex = featureIndex
        self.father_split_feature = father_split_feature
        self.father_split_feature_Categories = father_split_feature_Categories
        self.pvalues = pvalues
        self.qvalues = qvalues
        self.whereCond = whereCond
        self.whereCond_new = whereCond_new
        self.children = children

    def get_dict(self):
        child_dic = [] if self.children == None else (self.children[0].get_dict(), self.children[1].get_dict())
        return {
            "node_id": self.node_id,
            "nodeType": self.nodeType,
            "father_split_feature_Categories": self.father_split_feature_Categories,
            "count_ratio": self.count_ratio,
            "controlCount": self.controlCount,
            "treatedCount": self.treatedCount,
            "controlAvg": self.controlAvg,
            "treatedAvg": self.treatedAvg,
            "father_split_feature": self.father_split_feature,
            "pvalues": self.pvalues,
            "tau_i": self.prediction,

            #                 "node_order": self.node_order,
            #                 "nodePosition":self.nodePosition,
            #                 "tau_i_new":self.prediction_new,
            #                 "splitType": self.splitType,
            #                 "gain": self.gain,
            #                 "impurity": self.impurity,
            #                 "featureName": self.featureName,
            #                 "featureIndex": self.featureIndex,
            #                 "qvalues":self.qvalues,
            #                 "whereCond":self.whereCond,
            #                 "whereCond_new":self.whereCond_new,
            "children": child_dic}

    # causaltree_todict for tree.plot


class CTRegressionTree():
    def __init__(
            self,
            tree=0,
            total_count=1
    ):
        self.tree = tree
        self.total_count = total_count

    def get_decision_rules(self, node_order="root"):

        tree = self.tree
        node_id = tree.getTreeID()
        node_type = tree.get_node_type()
        node_order = node_order
        # 11.1 sort _Category values
        x = tree.father_split_feature_Categories
        #         x = [int(i) for i in x]
        x.sort()
        #         father_split_feature_Categories = [str(i) for i in x]

        if node_type == "internal":
            gain = tree.maxImpurGain
            feature_name = tree.splitFeat
            feature_index = tree.splitIndex
            split_type = "categorical"
            node_impurity = tree.impurity
            #             Categories = tree.featvalues_dict[feature_name]
            #             left_Categories = list(tree.splitIndex.split(','))
            #             right_Categories = []
            #             for i in Categories:
            #                 if i not in left_Categories:
            #                     right_Categories.append(i)
            left = CTRegressionTree(tree=tree.leftNode, total_count=self.total_count)
            right = CTRegressionTree(tree=tree.rightNode, total_count=self.total_count)
            children = (left.get_decision_rules("left"), right.get_decision_rules("right"))
        else:
            gain = None
            feature_name = None
            feature_index = None
            split_type = None
            node_impurity = None
            #             left_Categories = None
            #             right_Categories = None
            children = None

        ctDecisionNode = CTDecisionNode(
            node_id=node_id,
            nodeType=node_type,
            node_order=node_order,
            count_ratio=round(tree.inferenceSet['ratio'] * 100, 2),
            treatedCount=tree.inferenceSet['treatedCount'],
            controlCount=tree.inferenceSet['controlCount'],
            treatedAvg=round(tree.inferenceSet['treatedAvg'], 4),
            controlAvg=round(tree.inferenceSet['controlAvg'], 4),
            nodePosition=tree.nodePosition,
            prediction=round(tree.prediction, 4),
            prediction_new=[float(tree.inferenceSet['estPoint' + str(i)]) for i in range(1, 5)],
            splitType=split_type,
            gain=gain,
            impurity=node_impurity,
            featureName=feature_name,
            featureIndex=feature_index,
            father_split_feature=tree.father_split_feature,
            father_split_feature_Categories=tree.father_split_feature_Categories,
            pvalues=[round(float(tree.inferenceSet['pvalue' + str(i)]), 2) for i in range(1, 5)],
            qvalues=[round(float(tree.inferenceSet['qvalue' + str(i)]), 2) for i in range(1, 5)],
            children=children,
            whereCond=tree.whereCond,
            whereCond_new=tree.whereCond_new
        )
        return ctDecisionNode


def SelectSchema(schema):
    return schema


def FeatNames(schema):
    if schema == '':
        schemaArray = []
    elif '+' in schema:
        schemaArray = schema.split("+")
    else:
        schemaArray = [schema]
    return schemaArray


def FilterSchema(schemaArray):
    tmp = [i + " is not null" for i in schemaArray]
    return " and ".join(tmp)


def auto_wrap_text(text, max_line_length):
    wrapped_text = textwrap.fill(text, max_line_length)
    return wrapped_text


def check_numeric_type(table_, col):
    sql_instance = create_sql_instance()
    x = sql_instance.sql(f"desc {table_} ")
    cols_type = {}
    for i in range(len(x)):
        col_type = x[i]
        cols_type[col_type[0]] = col_type[1]

    if cols_type[col] not in ['UInt8', 'UInt16', 'UInt32', 'UInt64', 'UInt128', 'UInt256',
                              'Int8', 'Int16', 'Int32', 'Int64', 'Int128', 'Int256', 'Float32', 'Float64']:
        print(f"The type of {col} is not numeric")
        return 1



class CausalTree():
    def __init__(
            self,
            depth=3,
            min_sample_ratio_leaf = 0.001

    ):
        self.depth = depth
        self.threshold = min_sample_ratio_leaf
        self.Y = ''
        self.T = ''
        self.X = ''
        self.needcut_X = ''
        self.table = ''
        self.tree_structure = []
        self.result_df = []
        self.__sql_instance = create_sql_instance()
        self.__cutbinstring = ''
        self.feature_importance = pd.DataFrame([])

    def __params_input_check(self):

        sql_instance = self.__sql_instance

        if self.Y == '':
            print("missing Y. You should check out the input.")
            raise ValueError
        if self.T == '':
            print("missing T. You should check out the input.")
            raise ValueError
        if self.X == '':
            print("missing X. You should check out the input.")
            raise ValueError
        if self.table == '':
            print("missing table. You should check out the input.")
            raise ValueError
        else:
            sql_instance.sql(f"select {self.Y},{self.T},{self.X} from {self.table}")
            
            
    def __table_variables_check(self):

        sql_instance = self.__sql_instance
        table = self.variables['table']
        Y = self.variables['Y']
        T = self.variables['T']
        x_names = self.variables['x_names']
        cut_x_names = self.variables['cut_x_names']
        variables = Y + T + x_names + cut_x_names
        
        if 'Code: 60' in str(sql_instance.sql(f"desc {table}")[0][0]):
            print(f"table {table} doesn't exist")
            raise ValueError
            
        for i in variables:
            if 'Code: 47' in str(sql_instance.sql(f'select count({i})  from {table}')[0][0]):
                print(f"variable {i} can't be find in the table {table}")
                raise ValueError
        
        if check_numeric_type(table, Y[0]) == 1:
            print("Y must be numeric")
            raise ValueError
        if check_numeric_type(table, T[0]) == 1:
            print("T must be numeric")
            raise ValueError
                    
        if len(cut_x_names)!=0:
            for i in cut_x_names:
                if check_numeric_type(table, i) == 1:
                    print("needcut_X must be numeric to be cut bins by quantiles")
                    raise ValueError
        T_value = set(sql_instance.sql(f'select {T[0]} from  {table} group by {T[0]} limit 10'))
        if T_value != {(0,), (1,)}:
            print("The value of T must be either 0 or 1 ")
            raise ValueError
#         if len(T_value) < 2:
#             print("The value of T must have 0 and 1 ")
#             raise ValueError


    def fit(self, Y, T, X, needcut_X, table):

        sql_instance = self.__sql_instance
        table_new = f'{table}_{int(time.time())}_new'
        print(table_new)
        self.Y = Y
        self.T = T
        self.table = table
        self.X = X
        self.needcut_X = needcut_X
        depth = self.depth
        self.__params_input_check()
        
        x_names = list(set(FeatNames(X)))
        cut_x_names = list(set(FeatNames(needcut_X)))
        no_cut_x_names = list(set(x_names) - set(cut_x_names))
        cut_x_names_new = [i + '_buckets' for i in cut_x_names]
        x_names = no_cut_x_names + cut_x_names
        x_names_new = no_cut_x_names + cut_x_names_new
        featNames = x_names_new
        self.variables = {
                      'T':[T],
                      'Y':[Y],
                      'x_names':x_names,
                      'cut_x_names':cut_x_names,
                      'table':table}
        
        self.__table_variables_check()       

        # get bins for cut_x_names

        bins_dict = {}
        if len(cut_x_names_new) != 0:
            for i in cut_x_names:
#                 if check_numeric_type(table, i) == 1:
#                     print("needcut_X must be numeric to be cut bins by quantiles")
#                     raise ValueError

                result = sql_instance.sql(f'''select quantiles(0.25,0.5,0.75,0.9,0.95,0.99)({i}) 
                    from  {table}''')
                bins = result[0][0].replace('[', '').replace(']', '').split(',')

                if len(bins) == 0:
                    bins = [0]

                bins = list(np.sort(list(set([float(x) for x in bins]))))
                bins_dict[i] = bins
            strings = []
            for i in bins_dict:
                string = f'CutBins({i},{bins_dict[i]},False) as {i}_buckets'
                strings.append(string)
            cutbinstring = ','.join(strings) + ','
        else:
            cutbinstring = ''

        self.bins_dict = bins_dict
        
        for i in bins_dict:
            bins_dict[i] = [-float("inf")] + bins_dict[i] + [float("inf")]
        
        if len(no_cut_x_names)!=0:
            no_cut_x_names_string = ','.join([f'{i} as {i}' for i in no_cut_x_names]) + ','
        else:
            no_cut_x_names_string = ''
        self.__cutbinstring = cutbinstring
        self.__no_cut_x_names_string = no_cut_x_names_string
        clickhouse_create_view(clickhouse_view_name=table_new,
                               sql_statement=f"""
                 {Y} as Y, {T} as treatment,{cutbinstring}{no_cut_x_names_string} rand()%2 as if_test
          """,
                               sql_table_name=table,
                               bucket_column = 'if_test',
                               is_force_materialize=True)

        # check if empty data

        allcnt = sql_instance.sql(f'select count(*) from {table_new} where {FilterSchema(x_names_new)}')[0][0]
        #         print("total users:",allcnt)
        if allcnt == 0:
            print("Sample size is 0, check for null values")
            raise ValueError

        # check if T in (0,1) and Y is not string
        treatments = sql_instance.sql(f'select treatment from {table_new} group by treatment')
        treatments = set(np.array(treatments)[:, 0])
        if treatments != {0, 1}:
            print("The value of T can only be 0 or 1")
            raise ValueError

        if check_numeric_type(table_new, 'Y') == 1:
            print("Y must be numeric! ")
            raise ValueError

        # compute ate before build tree
        train = sql_instance.sql(
            f'SELECT if(treatment=1,1,-1) as z, sum(Y) as sum,count(*) as cnt FROM {table_new} where if_test=0 group by if(treatment=1,1,-1)')
        test = sql_instance.sql(
            f'SELECT if(treatment=1,1,-1) as z, sum(Y) as sum,count(*) as cnt FROM {table_new} where if_test=1 group by if(treatment=1,1,-1)')
        train = pd.DataFrame(train, columns=['z', 'sum', 'cnt'])
        test = pd.DataFrame(test, columns=['z', 'sum', 'cnt'])
        train['z'] = train['z'].astype(int)
        test['z'] = test['z'].astype(int)
        dat_cnt = train['cnt'].sum()
        estData_cnt = test['cnt'].sum()
        data_all = pd.merge(train, test, on='z')
        ate = (data_all['z'] * (data_all['sum_x'] + data_all['sum_y']) / (data_all['cnt_x'] + data_all['cnt_y'])).sum()
        estData_ate = ((test['z'] * (test['sum'])) / (test['cnt'])).sum()

        # build tree
        t1 = time.time()

        modelTree = CausalTreeclass(dfName=table_new,threshold=self.threshold, maxDepth=depth, whereCond='', nodePosition="L", impurity=0,
                                    father_split_feature="",
                                    father_split_feature_Categories=[], whereCond_new=[], nodesSet=[])
        modelTree.get_global_values(dat_cnt, depth, featNames)
        #         print("================================== start buildTree -- maxDepth: {maxDepth}, nodePosition: {nodePosition}==================================".format(maxDepth=depth,nodePosition="root"))
        modelTree.buildTree()
        #         print("============================================== build Tree Sucessfully=====================================================")
        result_list = []

        nodesSet = modelTree.nodesSet
        splitpoint_pdf_all = pd.DataFrame([],columns=['featName','featValue','splitpoint','ImpurityGain'])
        
        for l in nodesSet:
            l.predictNode(ate=estData_ate, cnt=estData_cnt)  # TODO
            result = l.inferenceSet
            splitpoint_pdf_all = pd.concat([splitpoint_pdf_all, l.splitpoint_pdf], axis=0)
            result_list.append(result)
            
        self.feature_importance = splitpoint_pdf_all.groupby(by=['featName'],as_index=False)['ImpurityGain'].sum()
        self.feature_importance = self.feature_importance.sort_values(by=['ImpurityGain'],ascending=False)
        self.feature_importance.columns = ['featName','importance']
        
        columns = ["TreeID", "level", "isLeaf", "whereCond", "whereCond_new", "ratio", "prediction", "treatedCount",
                   "controlCount",
                   "treatedAvg", "controlAvg", "treatedStd", "controlStd",
                   "estPoint1", "std1", "pvalue1", "zValue1", "lowerCI1", "upperCI1",
                   "estPoint2", "std2", "pvalue2", "zValue2", "lowerCI2", "upperCI2",
                   "estPoint3", "std3", "pvalue3", "zValue3", "lowerCI3", "upperCI3",
                   "estPoint4", "std4", "pvalue4", "zValue4", "lowerCI4", "upperCI4"]
        result_df = pd.DataFrame(result_list, columns=columns)
        #         print("============================================== compute Tree Sucessfully=====================================================")

        result_df['qvalue1'] = fdrcorrection(result_df['pvalue1'])[1]
        result_df['qvalue2'] = fdrcorrection(result_df['pvalue2'])[1]
        result_df['qvalue3'] = fdrcorrection(result_df['pvalue3'])[1]
        result_df['qvalue4'] = fdrcorrection(result_df['pvalue4'])[1]

        # for continous variable, give the cut bins mapping 
        # for example:  {"x_continuous_1_buckets":[4]} >>> {"x_continuous_1_buckets":'[89.62761587688087,95.04490061748047)'}

        Category_value_dicts = {}

        for i in x_names_new:

            try:
                values = list(modelTree.allsplitpoint_pdf[modelTree.allsplitpoint_pdf['featName'] == i]['featValue'])
                values = list(np.array(values))
                values.sort()    
            except:
                print(values)

            if i in cut_x_names_new:
                cut_bins_dict = {}
                bins = bins_dict[i.split("_buckets")[0]]
                intevals = []
                for j in range(len(bins) - 1):
                    intevals.append('[' + str(round(bins[j],4)) + ',' + str(round(bins[j + 1],4)) + ')')
                for value in values:
                    cut_bins_dict[value] = intevals[int(value) - 1]  # todo
                Category_value_dicts[i] = cut_bins_dict
            else:
                Category_value_dicts[i] = dict(zip(values, values))

        whereCond_new_list = []
        for i in range(1, len(nodesSet)):
            whereCond_new_ = {}
            for whereCond_new in result_df.loc[i, 'whereCond_new']:
                key = list(whereCond_new.keys())[0]
                value = list(whereCond_new.values())[0]
                value.sort()
                Category_value_dict = Category_value_dicts[key]
                whereCond_new_[key] = [Category_value_dict[j] for j in value]
            whereCond_new_list.append(whereCond_new_)
        result_df['whereCond_new'] = [''] + whereCond_new_list

        nodesSet[0].inferenceSet = dict(result_df.loc[0, :])
        for i in range(1, len(nodesSet)):
            nodesSet[i].inferenceSet = dict(result_df.loc[i, :])  # inferenceSet: list >> dict
            nodesSet[i].whereCond_new = result_df.loc[i, 'whereCond_new']
            Category_value_dict = Category_value_dicts[nodesSet[i].father_split_feature]
            father_split_feature_Categories = nodesSet[i].father_split_feature_Categories.copy()
            nodesSet[i].father_split_feature_Categories = [Category_value_dict[j] for j in
                                                           father_split_feature_Categories]

        # tree_structure
        ctRegressionTree = CTRegressionTree(tree=modelTree, total_count=test['cnt'].sum())  # estData total cnt
        tree_structure = ctRegressionTree.get_decision_rules("root")
        self.tree_structure = tree_structure
        self.result_df = result_df
        self.estimate = list(self.result_df['prediction'])
        self.estimate_stderr = list(self.result_df['std1'])
        self.pvalue = list(self.result_df['pvalue1'])
        self.estimate_interval = np.array((self.result_df[['lowerCI1', 'upperCI1']]))

        clickhouse_drop_view(clickhouse_view_name=table_new) 
        
    def __add_nodes_edges(self, tree, dot=None):
        if dot is None:
            dot = Digraph('g', filename='btree.gv',
                          node_attr={'shape': 'record', 'height': '.1', 'width': '1', 'fontsize': '9'})
            dot = Digraph()
            dot.node_attr.update(width='1', fontsize='9')
            node_id = int(tree['node_id'])
            node_id = f'Node : {node_id}'
            ate = f'CATE : {tree["tau_i"]}  (p_value:{tree["pvalues"][0]})'
            sample = f'sample size: control {tree["controlCount"]};  treatment {tree["treatedCount"]}'
            sample_ratio = f'sample_ratio:{tree["count_ratio"]}%'
            mean = f'mean: control {tree["controlAvg"]};  treatment {tree["treatedAvg"]}'

            dot.node(str(tree['node_id']), '\n'.join([node_id, sample_ratio, ate, sample, mean]))

        for child in tree['children']:
            if child:
                node_id = int(child['node_id'])
                node_id = f'Node : {node_id}'
                ate = f'CATE : {child["tau_i"]}  (p_value:{child["pvalues"][0]})'
                sample = f'sample size: control {child["controlCount"]};  treatment {child["treatedCount"]}'
                sample_ratio = f'sample_ratio:{child["count_ratio"]}%'
                mean = f'mean: control {child["controlAvg"]};  treatment {child["treatedAvg"]}'
                split_criterion = f'split_criterion:   {child["father_split_feature"]} in {child["father_split_feature_Categories"]}'

                dot.node(str(child['node_id']),
                         '\n'.join([node_id, auto_wrap_text(split_criterion, 60), sample_ratio, ate, sample, mean]))
                dot.edge(str(tree['node_id']), str(child['node_id']))
                self.__add_nodes_edges(child, dot)
        return dot

    def treeplot(self):
        tree_structure = self.tree_structure.get_dict()
        dot = self.__add_nodes_edges(tree_structure)

        return dot

    def hte_plot(self):
        # toc curve
        result_df = self.result_df
        toc_data_df = result_df[result_df['isLeaf'] == True][
            ['prediction', 'treatedCount', 'controlCount', 'treatedAvg', 'controlAvg']].sort_values(by='prediction',
                                                                                                    ascending=False)
        toc_data_df.reset_index(drop=True, inplace=True)
        toc_data_df['cnt'] = toc_data_df['treatedCount'] + toc_data_df['controlCount']
        toc_data_df['treatedsum'] = toc_data_df['treatedCount'] * toc_data_df['treatedAvg']
        toc_data_df['controlsum'] = toc_data_df['controlCount'] * toc_data_df['controlAvg']
        cnt = toc_data_df['cnt'].sum()
        toc_data_df['treatedcumsum'] = toc_data_df['treatedsum'].cumsum()
        toc_data_df['controlcumsum'] = toc_data_df['controlsum'].cumsum()
        toc_data_df['treatedcumcnt'] = toc_data_df['treatedCount'].cumsum()
        toc_data_df['controlcumcnt'] = toc_data_df['controlCount'].cumsum()
        toc_data_df['ratio'] = toc_data_df['cnt'] / cnt
        toc_data_df['ratio_sum'] = toc_data_df['ratio'].cumsum()
        toc_data_df['toc'] = toc_data_df['treatedcumsum'] / toc_data_df['treatedcumcnt'] - toc_data_df[
            'controlcumsum'] / toc_data_df['controlcumcnt']
        toc_data_df['qini'] = toc_data_df['treatedcumsum'] - toc_data_df['controlcumsum'] / toc_data_df[
            'controlcumcnt'] * toc_data_df['treatedcumcnt']
        toc_data_df['order'] = toc_data_df.index + 1
        toc_data_df['x'] = toc_data_df['ratio_sum']
        ate = list(toc_data_df['toc'])[-1]
        toc_data_df['toc1'] = ate
        toc_data_df['qini1'] = toc_data_df['x'] * list(toc_data_df['qini'])[-1]
        toc_data_df
        toc_data = toc_data_df[['x', 'toc', 'toc1', 'qini', 'qini1']]
        toc_data.columns = ['ratio', 'toc_tree', 'toc_random', 'qini_tree', 'qini_random']
        toc_data = toc_data.sort_values(by=['ratio'])
        
        result = toc_data
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 4.8))
        ax1.plot(result['ratio'], result['toc_tree'], label='CausalTree')
        ax2.plot([0] + list(result['ratio']), [0] + list(result['qini_tree']), label='CausalTree')
        ax1.plot(result['ratio'], result['toc_random'], label='Random Model')
        ax2.plot([0] + list(result['ratio']), [0] + list(result['qini_random']), label='Random Model')
        ax1.set_title('Cumulative Lift Curve')
        ax1.legend()
        ax2.set_title('Cumulative Gain Curve')
        ax2.legend()
        fig.suptitle('CausalTree Lift and Gain Curves')
        plt.show()

    def effect_2_clickhouse(self,table_output,table_input='',keep_col='*'):
        if table_input=='':
            table_input = self.table
        
        cutbinstring = self.__cutbinstring
        table_tmp = f'{table_input}_{int(time.time())}_foreffect_2_clickhouse'

        clickhouse_create_view(clickhouse_view_name=table_tmp, 
            sql_statement=f'''
                  *,{cutbinstring}1 as index
            ''', 
            sql_table_name = table_input, 
            bucket_column = 'index',
            is_force_materialize=True)
        
        leaf_effect = np.array(self.result_df[self.result_df['isLeaf']==True][['whereCond','prediction']])
        string = ' '.join([f'when True {x[0]} then {x[1]} ' for x in leaf_effect])

        clickhouse_create_view(clickhouse_view_name=table_output, 
            sql_statement=f'''
                 {keep_col},
                case 
                    {string}
                else 4294967295
                end as effect
                
            ''', 
            sql_table_name = table_tmp, 
            bucket_column = 'effect',
            is_force_materialize=True)
        
        clickhouse_drop_view(clickhouse_view_name=table_tmp) 
        