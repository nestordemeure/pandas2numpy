import warnings
import pandas as pd
import numpy as np

def assert_list_contains_all(l, l_subset):
    "Raise a warning if some columns from `l_subset` do not exist in `l`."
    non_existing_columns = set(l_subset).difference(l)
    if len(non_existing_columns) > 0:
        non_existing_columns_names = repr(list(non_existing_columns))
        warnings.warn("The columns " + non_existing_columns_names + " are not present in the dataframe and will be ignored!")

def list_intersection(l, l_superset):
    "Returns the intersection of `l` and `l_superset`."
    return list(set(l).intersection(l_superset))

class Pandas2numpy():
    "Dataframe to tensor converter for deep learning."
    def __init__(self, dataframe, continuous_columns=[], categorical_columns=[],
                       normalized_columns=[], NA_columns=[], logscale_columns=[]):
        """
        Stores information to be able to convert Pandas dataframe to Numpy array and back
        `dataframe` is an example dataframe used to determine all possible values in categories and store statistic for normalization and NA replacement
        `continuous_columns` is the name of the column containing continuous data to be encoded
        `categorical_columns` is the name of the columns containing categorical data to be encoded
        `normalized_columns` is the name of the columns that should be normalized by substracting the mean and dividing by the standard deviation
        `NA_columns` is the name of the coluns that might contain NA, cetgorical column will use an additional label while continuous column will replace NA with the median and store the presence of NA in an additional categorial column
        `logscale_columns` is the name of the columns to which a logarithm should be aplied (note that their elements should be strictly over 0)
        """
        # insures that all columns names are valid
        all_columns = dataframe.columns
        assert_list_contains_all(all_columns, continuous_columns)
        assert_list_contains_all(all_columns, categorical_columns)
        assert_list_contains_all(all_columns, normalized_columns)
        assert_list_contains_all(all_columns, logscale_columns)
        assert_list_contains_all(all_columns, NA_columns)
        # stores target column names
        self.continuous_columns = list_intersection(continuous_columns, all_columns)
        self.categorical_columns = list_intersection(categorical_columns, all_columns)
        self.normalized_columns = list_intersection(normalized_columns, self.continuous_columns)
        self.logscale_columns = list_intersection(logscale_columns, self.continuous_columns)
        self.NA_cont_columns = list_intersection(NA_columns, self.continuous_columns)
        self.NA_cat_columns = list_intersection(NA_columns, self.categorical_columns)
        # apply logscale transformation in order to measure normalization info in proper scale
        transformed_df = dataframe[self.continuous_columns]
        transformed_df.loc[:, self.logscale_columns] = transformed_df.loc[:, self.logscale_columns].apply(np.log)
        # stores normalization info
        self.normalized_columns_means = transformed_df[self.normalized_columns].mean(skipna=True)
        self.normalized_columns_std = transformed_df[self.normalized_columns].std(skipna=True)
        # stores median info for NA replacement
        self.NA_cont_columns_medians = dataframe[self.NA_cont_columns].median(skipna=True)
        # stores info on categories encoding
        self.category_dtypes = dataframe[self.categorical_columns].astype('category').dtypes
        # stores number of category per categorical column (useful to find embeding sizes and such)
        self.nb_category_per_categorical_column = []
        for (col_index, col_name) in enumerate(self.categorical_columns):
            # counts number of category per column
            nb_label = len(self.category_dtypes[col_index].categories)
            # adds one category when NA is a possibility
            if col_name in self.NA_cat_columns: nb_label += 1
            self.nb_category_per_categorical_column.append(nb_label)
        # adds the categorical columns created by NA_cont_columns
        for _ in range(len(self.NA_cont_columns)): self.nb_category_per_categorical_column.append(2) # 2 categories as those are booleans
        self.nb_category_per_categorical_column = np.array(self.nb_category_per_categorical_column)

    #--------------------------------------------------------------------------
    # ENCODING

    def continuous_to_numpy(self, df):
        """
        takes a dataframe and encode the `continuous_columns` as a tensor
        the NA in `NA_columns` are replaced with the medians of the columns in the example dataset
        takes the logarithm of the `logscale_columns` columns
        normalize the `normalized_columns` using a mean and standard deviation extracted from the example dataset
        
        `df` is the dataframe to be encoded
        """
        df = df[self.continuous_columns]
        # replace NA with median
        df.loc[:, self.NA_cont_columns] = df.loc[:, self.NA_cont_columns].fillna(self.NA_cont_columns_medians)
        # takes logarithm of some columns
        df.loc[:, self.logscale_columns] = df.loc[:, self.logscale_columns].apply(np.log)
        # normalizes some columns
        df.loc[:, self.normalized_columns] = (df.loc[:, self.normalized_columns] - self.normalized_columns_means) / self.normalized_columns_std
        return df.to_numpy()

    def categorial_to_numpy(self, df, include_continuous_NA_info=True):
        """
        takes a dataframe and encode the `categorical_columns` as a tensor of integers
        the NA in `NA_columns` are encoded as the 0 label of their respective columns
        addition columns are added to encode whether a continuous column in the `NA_columns` contained a NA
        `include_continuous_NA_info` should be set to true if you want additional columns encoding whether continuous variables contain NA
        """
        # encodes whether a cont column contained an NA (that was replaced by a median)
        if include_continuous_NA_info: continuous_col_isNA = df[self.NA_cont_columns].isna().astype(int)
        # encodes data as categories using predefined categories
        df = df[self.categorical_columns].astype(self.category_dtypes).apply(lambda x: x.cat.codes)
        # NA have code -1 by default, insures all codes are positive
        df.loc[:, self.NA_cat_columns] += 1
        # adds columns encoding whether continuous columns where NA
        if include_continuous_NA_info: df = pd.concat((df, continuous_col_isNA), axis=1)
        return df.to_numpy()

    def to_numpy(self, df, include_continuous_NA_info=True):
        """
        takes a dataframe and encode it as a pair `(tensor_cont,tensor_cat)`
        where `tensor_cont` stores the continuous columns
        and `tensor_cat` stores the categorial columns
        `include_continuous_NA_info` should be set to true if you want additional categorical columns encoding whether continuous variables contain NA
        """
        tensor_cont = self.continuous_to_numpy(df)
        tensor_cat = self.categorial_to_numpy(df, include_continuous_NA_info=include_continuous_NA_info)
        return (tensor_cont, tensor_cat)

    #--------------------------------------------------------------------------
    # DECODING

    def continuous_from_numpy(self, tensor_cont, copy=True):
        """
        takes a tensor and decodes it as a dataframe with columns `continuous_columns`
        removes the effects of normalization and logarithmic transformation
        you can set the `copy` argument to `False` to reduce memory usage but the decoding will modify your tensor in place
        WARNING: values that are notified as NA in the categorial tensor are left untouched here
        """
        df = pd.DataFrame(data=tensor_cont, columns=self.continuous_columns, copy=copy)
        # removes normalization
        df.loc[:, self.normalized_columns] = (df.loc[:, self.normalized_columns] * self.normalized_columns_std) + self.normalized_columns_means
        # removes logarithms
        df.loc[:, self.logscale_columns] = df.loc[:, self.logscale_columns].apply(np.exp)
        # nothing to do to reinject NAs at this stage
        return df

    def categorial_from_numpy(self, tensor_cat):
        """
        takes a tensor and decodes it as a dataframe with columns `categorical_columns`
        the columns encoding the presence of NA in continuous variables are ignored
        """
        columns = []
        # decodes one columns at a time
        # the columns encoding wether a cont column contained an NA are ignored
        for (col_index, col_name) in enumerate(self.categorical_columns):
            codes = tensor_cat[:,col_index]
            # gets NA back to code -1
            if col_name in self.NA_cat_columns: codes -= 1
            # translates codes in to their categories
            categories = self.category_dtypes[col_index].categories
            column = pd.Categorical.from_codes(codes, categories=categories)
            # save column
            columns.append(column)
        df = pd.DataFrame(data=columns, columns=self.categorical_columns)
        return df

    def from_numpy(self, tensor_cont, tensor_cat, copy=True, mask_NA=True):
        """
        takes a tensor representing the continuous variables and a tensor representing the categorical variables
        exports a dataframe
        if `mask_NA` is true, then the columns from `tensor_cat` encoding the presence of NA in continuous variables are used to replace those values with NA, otherwise they are left untouched
        you can set the `copy` argument to `False` to reduce memory usage but the decoding will modify `tensor_cont` in place
        """
        df_cont = self.continuous_from_numpy(tensor_cont, copy=copy)
        df_cat = self.categorial_from_numpy(tensor_cat)
        # use NA encoding column to replace continuous variables with NA
        if mask_NA:
            isNA_tensor = tensor_cat[:,len(self.categorical_columns):]
            isNA_df = pd.DataFrame(data=isNA_tensor, columns=self.NA_cont_columns).astype(bool)
            df_cont[isNA_df] = np.nan
        # merge dataframes
        df = pd.concat((df_cont, df_cat), axis=1)
        return df
