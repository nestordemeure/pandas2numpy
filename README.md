# Pandas2Numpy

Converting Pandas dataframes into Numpy tensor to help feeding them to deep learning frameworks.

## Usage

The class `Pandas2numpy` takes an aexample dataframe (that will be used to get metrics for later normalization), input and output column names plus names of columns to which various transformations will be applied.

Once build, it has an `encode` method that takes a dataframe and returns a triplet with the continuous, categorial and output columns transformed into three tensors.

It also has a `nb_label_per_category` member which stores the number of label per category (useful to make embeddings) in an array with one col per category.

```python
# a dataframe to be converted
df = ...

# builds encoder with an example dataframe to extract metrics for normalization
tabularEncoder = Pandas2numpy(df, continuous_columns=[], categorial_columns=[],
							      Normalized_columns=[], NA_columns=[], Log_columns=[])

# converts dataframe into pair of tensors
t_cont,t_cat = tabularEncoder.encode(df)

# converts subset of dataframe into pair of tensors
indexes_validationset = ...
t_cont,t_cat = tabularEncoder.encode(df[indexes_validationset]) # TODO

# converts row into pair of tensors
t_cont,t_cat = tabularEncoder.encode(df.iloc[0]) # TODO
```

*NOTE:* to encode outputs and inputs separately, you can just make one decoder for the inputs and another for the outputs.

-------------------------------------------------------------------------------------------------------

## categorial columns

columns are categorified if needed
stores encoding key to be able to encode new columns identically
converts them to long tensor

## Logarithm transformation
 
takes names of column containing values with large difference in magnitude
applies a logarithm to those columns

## NA transformation
 
takes names of columns containing NA
add a boolean contain_na col for each of those columns, include it in cat columns
replace NA by median value
 
## Normalize

normalizer:
take dataframe
return mean and std stored in rows to be used later
later we use those values

takes names of columns that should be normalized
substract a mean and divide by a standard deviation

-------------------------------------------------------------------------------------------------------

include `decode`, `decode_continuous` and `decode_categorial` methods

mostly useful when predicting in log space or a category
which needs decoding to get back to array values

`decode` uses `decode_continuous` and `decode_categorial`,
concatenates their outputs
adds the NA managing into account (turn values into NA were is_na is true, removes _is_na coluns)

-------------------------------------------------------------------------------------------------------

maybe also `encode_continuous`, `encode_categorial`
`categorial_to_numpy`
`categorial_from_numpy`
`to_numpy`

-------------------------------------------------------------------------------------------------------

To do deep-learning without fastai, it would be nice to have:
a function that displays a progress bar
a function that displays a convergence plot live

