def trainsampling(data,
                  samples=80,
                  col1='subject',
                  col2='classname'):
    """
    Function iterates through all unique combinations of two columns of a dataframe and pulls random samples for
    each combination equal to the number called in the 'samples' argument. Function will sample with replacement
    if the total number of rows per combination is less than the 'samples' argument. Samples will be returned
    as a pandas DataFrame.
    """

    # Raise error if selected columns are numeric
    if pd.api.types.is_numeric_dtype(data[col1]) or pd.api.types.is_numeric_dtype(data[col2]):
        raise TypeError('Columns must not be numeric')

    # Create empty dataframe
    dftemp = pd.DataFrame(columns=data.columns)

    # Assign list variables for unique values in each column
    col1ls = data.loc[:, col1].unique()
    col2ls = data.loc[:, col2].unique()

    # For loops to filter all combinations of the two columns and sample accordingly
    for i in col1ls:
        for j in col2ls:
            subset = data.loc[data.loc[:, col1] == i]
            subset = subset.loc[subset.loc[:, col2] == j]

            if len(subset) < samples:
                dftemp = pd.concat([dftemp, subset.sample(samples, replace=True)])

            else:
                dftemp = pd.concat([dftemp, subset.sample(samples, replace=False)])

    return dftemp