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
        self.logscale_columns = logscale_columns
        self.NA_cont_columns = list(set(NA_columns).intersection(continuous_columns))
        self.NA_cat_columns = list(set(NA_columns).intersection(categorial_columns))
        # apply logscale transformation in order to measure normalization info in proper scale
        transformed_df = dataframe[self.continuous_columns]
        transformed_df[self.logscale_columns] = transformed_df[self.logscale_columns].apply(np.log)
        # stores normalization info
        self.normalized_columns_means = transformed_df[self.normalized_columns].mean(skipna=True)
        self.normalized_columns_std = transformed_df[self.normalized_columns].std(skipna=True)
        # stores info on categories
        self.nb_label_per_category = 0 # TODO
    
    #--------------------------------------------------------------------------
    # ENCODING

    def continuous_to_numpy(self, df):
        df = df[self.continuous_columns]
        # replace NA with median
        df[self.NA_cont_columns] = df[self.NA_cont_columns].fillna(df[self.NA_cont_columns].median())
        # takes logarithm of some columns
        df[self.logscale_columns] = df[self.logscale_columns].apply(np.log)
        # normalizes some columns
        df[self.normalized_columns] = (df[self.normalized_columns] - self.normalized_columns_means) / self.normalized_columns_std
        return df.to_numpy()
    
    def categorial_to_numpy(self, df):
        # TODO use predefined categories
        # NA become -1 unless converted to string before cat?
        # could just add 1 to code in this case
        tensor = df[self.categorial_columns].astype('category') \
                                            .apply(lambda x: x.cat.codes) \
                                            .to_numpy()
        return tensor

    def to_numpy(self, df):
        tensor_cont = self.continuous_to_numpy(df)
        tensor_cat = self.categorial_to_numpy(df)
        return (tensor_cont, tensor_cat)
    
    #--------------------------------------------------------------------------
    # DECODING

    def continuous_from_numpy(self, tensor_cont, copy=True):
        """WARNING: this does not take NA into account
        use copy to avoid side effects"""
        df = pd.DataFrame(data=tensor_cont, columns=self.continuous_columns, copy=copy)
        # removes normalization
        df[self.normalized_columns] = (df[self.normalized_columns] * self.normalized_columns_std) + self.normalized_columns_means
        # removes logarithms
        df[self.logscale_columns] = df[self.logscale_columns].apply(np.exp)
        # nothing to do to reinject NAs at this stage
        return df
    
    def categorial_from_numpy(self, tensor_cat, copy=True):
        """"use copy to avoid side effects"""
        df = pd.DataFrame(data=tensor_cat, columns=self.categorial_columns, copy=copy)
        # TODO removes NA columns
        # TODO converts from code to categories
        # pd.Categorical.from_codes(codes, categories=["train", "test"])
        return df

    def from_numpy(self, tensor_cont, tensor_cat, copy=True):
        """"use copy to avoid side effects"""
        df_cont = self.continuous_from_numpy(tensor_cont, copy=copy)
        df_cat = self.categorial_from_numpy(tensor_cat, copy=copy)
        # merge dataframes
        df = pd.concat((df_cont, df_cat), axis=1)
        # TODO take care of cont NA columns
        return df
