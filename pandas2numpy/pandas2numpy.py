import pandas as pd
import numpy as np

class Pandas2numpy():
    "Dataframe to tensor converter for deep learning."
    def __init__(self, dataframe, continuous_columns=[], categorial_columns=[],
                       normalized_columns=[], NA_columns=[], logscale_columns=[]):
        # stores target column names
        self.continuous_columns = continuous_columns
        self.categorial_columns = categorial_columns
        self.normalized_columns = normalized_columns
        self.NA_columns = NA_columns
        self.logscale_columns = logscale_columns
        # stores info on continuous data
        # TODO
        # stores info on categories
        self.nb_label_per_category = 0 # TODO
    
    #--------------------------------------------------------------------------
    # ENCODING

    def continuous_to_numpy(self, df):
        return 0 # TODO
    
    def categorial_to_numpy(self, df):
        return 0 # TODO

    def to_numpy(self, df):
        tensor_cont = self.continuous_to_numpy(df)
        tensor_cat = self.categorial_to_numpy(df)
        return (tensor_cont, tensor_cat)
    
    #--------------------------------------------------------------------------
    # DECODING

    def continuous_from_numpy(self, tensor_cont):
        return 0 # TODO
    
    def categorial_from_numpy(self, tensor_cat):
        return 0 # TODO

    def from_numpy(self, tensor_cont, tensor_cat):
        df_cont = self.continuous_from_numpy(tensor_cont)
        df_cat = self.categorial_from_numpy(tensor_cat)
        # merge dataframes
        df = pd.concat((df_cont, df_cat), axis=1)
        # TODO take care of NA columns
        return df
