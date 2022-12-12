import logging
import types

import numpy as np
import pandas as pd
from sklearn import model_selection


class CleanAndPreprocess:

    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def lower_case(self, columns=[]):
        self.dataframe[columns] = self.dataframe[columns].apply(lambda x: x.str.lower(), axis=1)

    def remove_from_rows(self, columns=[], removals=[], delimiter=' '):
        if type(columns) is list:
            for col_name in columns:
                temp_lambda = lambda row: self.remove_from_text(row, col_name, removals=removals, delimiter=delimiter)
                self.dataframe = self.dataframe.apply(temp_lambda, axis=1)
        else:
            logging.warning(
                "remove_from_rows is expecting the variable columns to be of type list but you passed " + str(
                    type(columns))
            )

    def replace_in_rows(self, columns=[], replacements=[], delimiter=' '):
        if type(columns) is list:
            for col_name in columns:
                temp_lambda = lambda row: self.replace_in_text(row, col_name, replacements=replacements,
                                                               delimiter=delimiter)
                self.dataframe = self.dataframe.apply(temp_lambda, axis=1)
        else:
            logging.warning(
                "replace_in_rows is expecting the variable columns to be of type list but you passed " + str(
                    type(columns))
            )

    def remove_from_text(self, row, col_name, removals=[], delimiter=' '):
        replacements = [(r, '') for r in removals]
        text = self.replace_in_text(row, col_name, replacements=replacements, delimiter=delimiter)
        return text

    def replace_in_text(self, row, col_name, replacements=[], delimiter=' '):
        text = row[col_name]

        # replacements is a list of 2-tuples where the each entry is either a symbol or lambda
        if len(delimiter) > 0:
            elements = text.split(delimiter)
        else:
            elements = text

        split_text = []
        for elem in elements:

            if(len(elem) == 0):
                continue

            symbol_to_use = elem
            for r in replacements:
                if type(r[0]) is types.LambdaType:
                    if r[0](elem):
                        if type(r[1]) is types.LambdaType:
                            symbol_to_use = r[1](elem)
                        else:
                            symbol_to_use = r[1]

                        break
                elif elem == r[0]:
                    if type(r[1]) is types.LambdaType:
                        symbol_to_use = r[1](elem)
                    else:
                        symbol_to_use = r[1]
                    break

            if len(symbol_to_use) > 0:
                split_text.append(delimiter)

            split_text.append(symbol_to_use)

        if not text.startswith(delimiter):
            split_text[0] = ""

        text = ''.join(split_text)

        row[col_name] = text

        return row

    def create_stratified_folds(self, num_folds=5):
        # create a new column called kfold and fill it with -1
        self.dataframe['kfold'] = -1
        # randomize the rows of the data
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        # calculate the number of bins by Sturge's rule
        num_bins = int(np.floor(1 + np.log2(len(self.dataframe))))
        # define the bins for the stratified folds
        self.dataframe.loc[:, "bins"] = pd.cut(self.dataframe["target"], bins=num_bins, labels=False)
        kf = model_selection.StratifiedKFold(n_splits=num_folds)
        for f, (t_, v_) in enumerate(kf.split(X=self.dataframe, y=self.dataframe.bins.values)):
            self.dataframe.loc[v_, 'kfold'] = f
        self.dataframe = self.dataframe.drop("bins", axis=1)




