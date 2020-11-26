import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


def cvmodeleval(model,
                data,
                itercol='subject',
                n_iterations=1,
                epochs=1,
                batch_size=32,
                steps_per_epoch=None,
                validation_steps=None,
                target_size=(227, 227),
                random_state=None,
                patience=3):
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
    wsave = model.get_weights()

    # Instantiate image data generator
    datagen = keras.preprocessing.image.ImageDataGenerator()

    # Designate model checkpoint and callbacks_list
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 mode='max',
                                 monitor='val_accuracy',
                                 save_best_only=True)

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience)

    callbacks_list = [checkpoint, earlystop]

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
            model.set_weights(wsave)

            # Split train and test sets, iterating through each subject to be excluded from training
            cvtrain = data.loc[data.loc[:, itercol] != j]
            cvtest = data.loc[data.loc[:, itercol] == j]

            # Split training data
            train = datagen.flow_from_dataframe(cvtrain,
                                                x_col='imgpath',
                                                y_col='classname',
                                                batch_size=batch_size,
                                                target_size=target_size,
                                                seed=random_state)
            # Split validation data
            val = datagen.flow_from_dataframe(cvtest,
                                              x_col='imgpath',
                                              y_col='classname',
                                              target_size=target_size,
                                              seed=random_state)
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

    # Fill dataframe with stats
    dftemp = pd.DataFrame({'iteration': iterations, 'validation_subject': validation_subjects,
                           'validation_accuracy': validation_accuracies})

    return dftemp


def samplecv(model,
             data,
             isampled=1,
             samples=80,
             col1='subject',
             col2='classname',
             itercol='subject',
             n_iterations=1,
             epochs=3,
             batch_size=32,
             steps_per_epoch=None,
             validation_steps=None,
             target_size=(227, 227),
             random_state=None,
             patience=3):
    """
    Combine trainsampling and cvmodeleval functions so that the training data can be resampled numerous times
    and run through cvmodeleval, to get a better representation of the model's performance.
    """

    modelsave = model

    for k in range(isampled):

        # Save and print out iteration info
        fullit = str(k + 1)
        print('Resample iteration ' + fullit)

        # Perform training sampling
        ts = trainsampling(data=data, samples=samples, col1=col1, col2=col2, random_state=random_state)

        # Run CV model evaluation
        stats = cvmodeleval(model=modelsave, data=ts, itercol=itercol, n_iterations=n_iterations, epochs=epochs,
                            batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                            target_size=target_size, random_state=random_state, patience=patience)

        if k == 0:
            combinedstats = stats

            combinedstats.rename(columns={'validation_accuracy': 'val_acc1'}, inplace=True)

        else:
            # noinspection PyUnboundLocalVariable
            combinedstats['val_acc' + fullit] = stats['validation_accuracy']

    return combinedstats


def trainsampling(data,
                  samples=80,
                  col1='subject',
                  col2='classname',
                  random_state=None):
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
                dftemp = pd.concat([dftemp, subset.sample(samples, replace=True, random_state=random_state)])

            else:
                dftemp = pd.concat([dftemp, subset.sample(samples, replace=False, random_state=random_state)])

    return dftemp


def cvrand(model,
           data,
           traindatagen,
           testdatagen,
           itercol='subject',
           n_iterations=3,
           val_subjects=3,
           epochs=1,
           batch_size=32,
           steps_per_epoch=None,
           validation_steps=None,
           target_size=(227, 227),
           random_state=None,
           min_delta=0,
           patience=3):
    """
    Define function to perform cross-validation on a model. Function will split the data into training and validation
    sets by randomly selecting a defined number of validation_subjects (without replacement) and putting their data
    in the validation set for that iteration. The goal is to build a model that generalizes well to all subjects,
    and doesn't just remember the specific subjects present in the training set.

    The model will be reset to default parameter weights each iteration and will be fit to the cross-validation data.
    The maximum validation_accuracy achieved (as well as the associated training accuracy during that specific epoch)
    from each iteration of fitting will be logged into a pandas DataFrame.
    """

    # Raise error if selected columns are numeric
    if pd.api.types.is_numeric_dtype(data[itercol]):
        raise TypeError('Columns must not be numeric')

    # set random seed
    np.random.seed(random_state)

    # Create empty lists
    sampledvalues = []
    validation_accuracies = []
    train_accuracies = []

    # Save initial default model weights
    wsave = model.get_weights()

    # Designate model checkpoint and callbacks_list
    checkpoint = ModelCheckpoint('weights.hdf5',
                                 mode='max',
                                 monitor='val_accuracy',
                                 save_best_only=True)

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)

    callbacks_list = [checkpoint, earlystop]

    # Initialize iteration counter
    counter = 0

    # Pull unique values from itercol
    valuelist = data[itercol].unique()

    for i in range(n_iterations):
        sampledvalues.append(np.random.choice(valuelist, size=val_subjects, replace=False))

    # n_iteration for loop
    for i in range(n_iterations):
        # Substep counter
        counter += 1

        # Print iteration and substep progress
        print('CV iteration ' + str(counter) + ' of ' + str(n_iterations))

        # reset model states for fresh training
        model.set_weights(wsave)

        # Sample the validation values
        print('Validation subjects are ' + str(sampledvalues[i]))

        # Split train and test sets, iterating through each subject to be excluded from training
        cvtest = data[data[itercol].isin(sampledvalues[i])]
        cvtrain = data[~data[itercol].isin(sampledvalues[i])]

        # Split training data
        train = traindatagen.flow_from_dataframe(cvtrain,
                                                 x_col='imgpath',
                                                 y_col='classname',
                                                 batch_size=batch_size,
                                                 target_size=target_size,
                                                 seed=random_state)
        # Split validation data
        val = testdatagen.flow_from_dataframe(cvtest,
                                              x_col='imgpath',
                                              y_col='classname',
                                              target_size=target_size,
                                              seed=random_state)
        # Fit model
        model.fit(train,
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val,
                  validation_steps=validation_steps,
                  callbacks=callbacks_list)

        # Append lists
        valmax = max(model.history.history['val_accuracy'])
        valmaxindex = model.history.history['val_accuracy'].index(valmax)
        validation_accuracies.append(round(valmax, 3))
        train_accuracies.append(round(model.history.history['accuracy'][valmaxindex], 3))

    # Fill dataframe with stats
    dftemp = pd.DataFrame({'validation_subjects': sampledvalues,
                           'train_accuracies': train_accuracies,
                           'validation_accuracy': validation_accuracies})

    return dftemp
