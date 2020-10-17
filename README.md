# Pandas2Numpy

Converting Pandas dataframes into Numpy tensor to help feeding them to deep learning frameworks.
This library tries to be modular, easy to use with any deep-learning framework by building on the common numpy API, and non-surprising in its behaviour.

## Instalation

You can install our librarie with:

```
pip install git+https://github.com:nestordemeure/pandas2numpy.git
```

## Usage

The `Pandas2numpy` class takes an example dataframe and column names in order to build an object that can encode/decode dataframe properly.
It also takes information on columns you might want to normalize, put in logscale and that might contain NA that should be dealt with.

NA in categorical columns are encoded by adding an additional category.
NA in continuous columns are replaced by the median value of the column (as computed in the example dataframe) and marked in a dedicated categorical column.

```python
from pandas2numpy import Pandas2numpy

# example dataframe
df = pandas.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# continuous variables to be encoded
continuous_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# categorical_columns to be encoded
categorical_columns = ['species']
# columns that should be set to mean=0, std=1
normalized_columns = ['sepal_length', 'sepal_width']
# columns that might contain NA
NA_columns = ['sepal_width', 'species']
# columns to which a logarithm should be applied
logscale_columns = ['sepal_length', 'petal_length']

# builds an encoder with an example dataframe to extract metrics for normalization and possible categories
tabularEncoder = Pandas2numpy(df, continuous_columns=continuous_columns, categorical_columns=categorical_columns,
                              normalized_columns=normalized_columns, NA_columns=NA_columns, logscale_columns=logscale_columns)
```

Once constructed, you can use its `to_numpy` methods to convert dataframes and rows into numpy tensors.
We also provide methods that deal with categorical and continuous variables only.

```python
# converts a dataframe into a tensor of floats and a tensor of ints
tensor_continuous,tensor_categorical = tabularEncoder.to_numpy(df)

# converts only continuous data into a tensor
tensor_continuous2 = tabularEncoder.continuous_to_numpy(df)

# converts a row (only the categorical data in this example)
# note the `df.iloc[[0]]` syntax to ensure that the row is in a dataframe and not a serie
tensor_categorical2 = tabularEncoder.categorial_to_numpy(df.iloc[[0]])
```

The `from_numpy` methods convert arrays back to dataframes.

```python
# converts tensors back into a dataframe (note that the order of columns might change)
df = tabularEncoder.from_numpy(tensor_continuous,tensor_categorical)

# converts the continuous tensor back into a dataframe (that will only include continuous columns)
df_continuous2 = tabularEncoder.continuous_from_numpy(tensor_continuous2)

# converts the categorical tensor into a one row dataframe
row_categorical2 = tabularEncoder.categorial_from_numpy(tensor_categorical2)
```

`Pandas2numpy` also has a `nb_category_per_categorical_column` member containing a numpy array with the number of category encoded as int for each column of the categorical tensor (which is useful to make embeddings).

```python
nb_categories = tabularEncoder.nb_category_per_categorical_column
```

For further information, we invite you to read the [documentation of the individual functions](https://github.com/nestordemeure/pandas2numpy/blob/main/pandas2numpy/pandas2numpy.py).
