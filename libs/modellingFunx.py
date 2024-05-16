import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BusinessDecisionTreeClassifier(BaseEstimator, TransformerMixin):
    """Constructed Decision Tree Classifier based on segmentation"""
    def __init__(self,
                 get_nodes = True,
                 debt_name = 'DEUDA',
                 liabilities_name = 'MEDIANA_AHORROS_ULT_6M',
                 income_name = 'INGRESO_CLIENTE',
                 cluster_name = 'CLUSTER',
                 node_name = 'NODE',
                 rare_cluster = 999,
                 rare_node = 999):
        
        self.get_nodes = get_nodes
        self.debt_name = debt_name
        self.liabilities_name = liabilities_name
        self.income_name = income_name
        self.cluster_name = cluster_name
        self.node_name = node_name
        self.rare_cluster = rare_cluster
        self.rare_node = rare_node


    def __create_segments(self):

        """Cutoffs for decision tree classifier"""

        nodes = {1: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] <= 8800) &\
                              (x[self.income_name] <= 3500),
                              
                 2: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] <= 8800) &\
                              (x[self.income_name] > 3500),
                               
                 3: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] > 700) &\
                              (x[self.income_name] <= 3000),

                 4: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] <= 700),

                 5: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] > 3000),

                 6: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] > 1650) &\
                              (x[self.income_name] <= 2650),

                 7: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] <= 1650),

                 8: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] > 2650),

                 9: lambda x: (x[self.debt_name] > 7500) &\
                              (x[self.debt_name] <= 9500) &\
                              (x[self.liabilities_name] <= 5200) &\
                              (x[self.income_name] > 900) &\
                              (x[self.income_name] <= 3000),

                 10: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] <= 5200) &\
                               (x[self.income_name] <= 900),

                 11: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] <= 5200) &\
                               (x[self.income_name] > 3000),

                 12: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] > 650) &\
                               (x[self.income_name] <= 2800),

                 13: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] <= 650),                            

                 14: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] > 2800),

                 15: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] > 1450) &\
                               (x[self.income_name] <= 2650),

                 16: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] <= 1450),

                 17: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] > 2650),

                 18: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] > 1100) &\
                               (x[self.income_name] <= 2600),

                 19: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] <= 1100),

                 20: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] > 2600),

                 21: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] > 7000)}

        # declare cluster
        clusters = {0: lambda x: (x[self.node_name] == 1) |\
                                 (x[self.node_name] == 3) |\
                                 (x[self.node_name] == 6) |\
                                 (x[self.node_name] == 9) |\
                                 (x[self.node_name] == 12) |\
                                 (x[self.node_name] == 15) |\
                                 (x[self.node_name] == 18) |\
                                 (x[self.node_name] == 21),

                    1: lambda x: (x[self.node_name] == 2) |\
                                 (x[self.node_name] == 4) |\
                                 (x[self.node_name] == 5) |\
                                 (x[self.node_name] == 7) |\
                                 (x[self.node_name] == 8) |\
                                 (x[self.node_name] == 10) |\
                                 (x[self.node_name] == 11) |\
                                 (x[self.node_name] == 13) |\
                                 (x[self.node_name] == 14) |\
                                 (x[self.node_name] == 16) |\
                                 (x[self.node_name] == 17) |\
                                 (x[self.node_name] == 19) |\
                                 (x[self.node_name] == 20)}
        
        return nodes, clusters
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X = X.copy()

        nodes, clusters = self.__create_segments()

        X[self.node_name] = np.select([node(X) for node in nodes.values()],
                                      list(nodes.keys()),
                                      default = self.rare_node)\
                              .astype('int16')
        
        X[self.cluster_name] = np.select([cluster(X) for cluster in clusters.values()],
                                         list(clusters.keys()),
                                         default = self.rare_cluster)\
                                 .astype('int16')
        
        if self.get_nodes == False:
            X.drop(columns = self.node_name, inplace = True)

        return X
    
class FixedBusinessDecisionTreeClassifier(BaseEstimator, TransformerMixin):
    """Constructed Decision Tree Classifier based on segmentation"""
    def __init__(self,
                 get_nodes = True,
                 debt_name = 'DEUDA',
                 liabilities_name = 'MEDIANA_AHORROS_ULT_6M',
                 income_name = 'INGRESO_CLIENTE',
                 cluster_name = 'FIXED_CLUSTER',
                 node_name = 'NODE',
                 rare_cluster = 999,
                 rare_node = 999):
        
        self.get_nodes = get_nodes
        self.debt_name = debt_name
        self.liabilities_name = liabilities_name
        self.income_name = income_name
        self.cluster_name = cluster_name
        self.node_name = node_name
        self.rare_cluster = rare_cluster
        self.rare_node = rare_node


    def __create_segments(self):

        """Cutoffs for decision tree classifier"""

        nodes = {1: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] <= 8800) &\
                              (x[self.income_name] <= 3500),
                              
                 2: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] <= 8800) &\
                              (x[self.income_name] > 3500),
                               
                 3: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] > 700) &\
                              (x[self.income_name] <= 3000),

                 4: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] <= 700),

                 5: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 8800) &\
                              (x[self.liabilities_name] <= 12000) &\
                              (x[self.income_name] > 3000),

                 6: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] > 1650) &\
                              (x[self.income_name] <= 2650),

                 7: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] <= 1650),

                 8: lambda x: (x[self.debt_name] <= 7500) &\
                              (x[self.liabilities_name] > 12000) &\
                              (x[self.income_name] > 2650),

                 9: lambda x: (x[self.debt_name] > 7500) &\
                              (x[self.debt_name] <= 9500) &\
                              (x[self.liabilities_name] <= 5200) &\
                              (x[self.income_name] > 900) &\
                              (x[self.income_name] <= 3000),

                 10: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] <= 5200) &\
                               (x[self.income_name] <= 900),

                 11: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] <= 5200) &\
                               (x[self.income_name] > 3000),

                 12: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] > 650) &\
                               (x[self.income_name] <= 2800),

                 13: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] <= 650),                            

                 14: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 5200) &\
                               (x[self.liabilities_name] <= 8800) &\
                               (x[self.income_name] > 2800),

                 15: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] > 1450) &\
                               (x[self.income_name] <= 2650),

                 16: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] <= 1450),

                 17: lambda x: (x[self.debt_name] > 7500) &\
                               (x[self.debt_name] <= 9500) &\
                               (x[self.liabilities_name] > 8800) &\
                               (x[self.income_name] > 2650),

                 18: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] > 1100) &\
                               (x[self.income_name] <= 2600),

                 19: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] <= 1100),

                 20: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] <= 7000) &\
                               (x[self.income_name] > 2600),

                 21: lambda x: (x[self.debt_name] > 9500) &\
                               (x[self.liabilities_name] > 7000)}

        # declare cluster
        clusters = {'PEAK': lambda x: (x[self.node_name] == 5) |\
                                      (x[self.node_name] == 6) |\
                                      (x[self.node_name] == 7) |\
                                      (x[self.node_name] == 8) |\
                                      (x[self.node_name] == 12) |\
                                      (x[self.node_name] == 14) |\
                                      (x[self.node_name] == 15) |\
                                      (x[self.node_name] == 16) |\
                                      (x[self.node_name] == 17) |\
                                      (x[self.node_name] == 20) |\
                                      (x[self.node_name] == 21),

                    'CORPUS': lambda x: (x[self.node_name] == 1) |\
                                        (x[self.node_name] == 2) |\
                                        (x[self.node_name] == 3) |\
                                        (x[self.node_name] == 4) |\
                                        (x[self.node_name] == 9) |\
                                        (x[self.node_name] == 10) |\
                                        (x[self.node_name] == 11) |\
                                        (x[self.node_name] == 13) |\
                                        (x[self.node_name] == 18) |\
                                        (x[self.node_name] == 19)}
        
        return nodes, clusters
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X = X.copy()

        nodes, clusters = self.__create_segments()

        X[self.node_name] = np.select([node(X) for node in nodes.values()],
                                      list(nodes.keys()),
                                      default = self.rare_node)\
                              .astype('int16')
        
        X[self.cluster_name] = np.select([cluster(X) for cluster in clusters.values()],
                                         list(clusters.keys()),
                                         default = self.rare_cluster)\
                                 .astype('object')
        
        if self.get_nodes == False:
            X.drop(columns = self.node_name, inplace = True)

        return X