import pandas as pd
import numpy as pd
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