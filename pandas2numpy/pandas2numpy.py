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
        # stores info on categories encoding
        self.category_dtypes = dataframe[self.categorial_columns].astype('category').dtypes
        # stores number of label per category (useful to find embeding sizes and such)
        self.nb_label_per_category = []
        for (col_index, col_name) in enumerate(self.categorial_columns):
            # counts number of label per column
            nb_label = len(self.category_dtypes[col_index].categories)
            # adds one label when NA is a possibility
            if col_name in self.NA_cat_columns: nb_label += 1
            self.nb_label_per_category.append(nb_label)

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
        """adds categorial column to encode wether continuous NA_columns columns contains NA"""
        # encodes whether a cont column contained an NA (that was replaced by a median)
        continuous_col_isNA = df[self.NA_cont_columns].isna().astype(int)
        # encodes data as categories using predefined categories
        df = df[self.categorial_columns].astype(self.category_dtypes).apply(lambda x: x.cat.codes)
        # NA have code -1 by default, insures all codes are positive
        df[self.NA_cat_columns] += 1
        # adds columns encoding whether continuous columns where NA
        df = pd.concat((df, continuous_col_isNA), axis=1)
        return df.to_numpy()

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

    def categorial_from_numpy(self, tensor_cat):
        columns = []
        # decodes one columns at a time
        # the columns encoding wether a cont column contained an NA are ignored
        for (col_index, col_name) in enumerate(self.categorial_columns):
            codes = tensor_cat[:,col_index]
            # gets NA back to code -1
            if col_name in self.NA_cat_columns: codes -= 1
            # translates codes in to their categories
            categories = self.category_dtypes[col_index].categories
            column = pd.Categorical.from_codes(codes, categories=categories)
            # save column
            columns.append(column)
        df = pd.DataFrame(data=columns, columns=self.categorial_columns)
        return df

    def from_numpy(self, tensor_cont, tensor_cat, copy=True, decode_NA=True):
        """use copy to avoid side effects"""
        df_cont = self.continuous_from_numpy(tensor_cont, copy=copy)
        df_cat = self.categorial_from_numpy(tensor_cat)
        # use NA encoding column to replace continuous variables with NA
        if decode_NA:
            isNA_tensor = tensor_cat[:,len(self.categorial_columns):]
            isNA_df = pd.DataFrame(data=isNA_tensor, columns=self.NA_cont_columns).astype(bool)
            df_cont[isNA_df] = np.nan
        # merge dataframes
        df = pd.concat((df_cont, df_cat), axis=1)
        return df
