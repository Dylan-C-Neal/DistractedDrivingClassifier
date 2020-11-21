def cvmodeleval(model,
                data,
                itercol='subject',
                n_iterations=1,
                epochs=1,
                steps_per_epoch=None,
                validation_steps=None,
                target_size=(227, 227)):
    """
    Define function to perform cross-validation on a model. Function will split the data into training and validation
    sets by each unique value in the itercol. In this case the itercol represents unique subjects present in each
    photo. The goal is to build a model that generalizes well to all subjects, and doesn't just remember the specific
    subjects present in the training set.

    The model will be reset to default parameter weights each iteration and will be fit to the cross-validation data.
    The maximum validation_accuracy achieved during each round of fitting will be logged into a pandas DataFrame.

    n_iterations represents the total number of times the full iteration loop will run, thus allowing multiple
    data points to be collected for each unique value in the itercol.
    """

    # Raise error if selected columns are numeric
    if pd.api.types.is_numeric_dtype(data[itercol]):
        raise TypeError('Columns must not be numeric')

    # Create empty lists
    iterations = []
    validation_subjects = []
    validation_accuracies = []

    # Save initial default model weights
    Wsave = model.get_weights()

    # Instantiate image data generator
    datagen = keras.preprocessing.image.ImageDataGenerator()

    # Designate model checkpoint and callbacks_list
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 mode='max',
                                 monitor='val_accuracy',
                                 save_best_only=True)

    callbacks_list = [checkpoint]

    # n_iteration for loop
    for i in range(n_iterations):

        # Initialize Substep counter
        counter = 0

        for j in data[itercol].unique():
            # Substep counter
            counter += 1

            # Print iteration and substep progress
            print('CV iteration ' + str(i + 1))
            print('Substep ' + str(counter) + ' of ' + str(len(data[itercol].unique())))

            # reset model states for fresh training
            model.set_weights(Wsave)

            # Split train and test sets, iterating through each subject to be excluded from training
            cvtrain = data.loc[data.loc[:, 'subject'] != j]
            cvtest = data.loc[data.loc[:, 'subject'] == j]

            # Split training data
            train = datagen.flow_from_dataframe(cvtrain,
                                                x_col='imgpath',
                                                y_col='classname',
                                                target_size=target_size)
            # Split validation data
            val = datagen.flow_from_dataframe(cvtest,
                                              x_col='imgpath',
                                              y_col='classname',
                                              target_size=target_size)
            # Fit model
            model.fit(train,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=val,
                      validation_steps=validation_steps,
                      callbacks=callbacks_list)

            # Append lists
            iterations.append(i + 1)
            validation_subjects.append(j)
            validation_accuracies.append(round(max(model.history.history['val_accuracy']), 3))

            # Clear tf backend
            tf.keras.backend.clear_session()

    # Fill dataframe with stats
    dftemp = pd.DataFrame({'iteration': iterations, 'validation_subject': validation_subjects,
                           'validation_accuracy': validation_accuracies})

    return dftemp


def sampleCV(model,
             data,
             isampled=3,
             samples=80,
             col1='subject',
             col2='classname',
             itercol='subject',
             n_iterations=1,
             epochs=1,
             steps_per_epoch=None,
             validation_steps=None,
             target_size=(227, 227)):
    """
    Combine trainsampling and cvmodeleval functions so that the training data can be resampled numerous times
    and run through cvmodeleval, to get a better representation of the model's performance.
    """

    for k in range(isampled):

        # Save and print out iteration info
        fullit = str(k + 1)
        print('Resample iteration ' + fullit)

        # Perform training sampling
        ts = trainsampling(data=data, samples=samples, col1=col1, col2=col2)

        # Save input model as separate variable, so that learned model parameter weights don't get logged for next
        # iteration
        modelsave = model

        # Run CV model evaluation
        stats = cvmodeleval(model=modelsave, data=ts, itercol=itercol, n_iterations=n_iterations, epochs=epochs,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, target_size=target_size)

        if k == 0:
            combinedstats = stats

            combinedstats.rename(columns={'validation_accuracy': 'val_acc1'}, inplace=True)

        else:
            combinedstats['val_acc' + fullit] = stats['validation_accuracy']

    return combinedstats


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