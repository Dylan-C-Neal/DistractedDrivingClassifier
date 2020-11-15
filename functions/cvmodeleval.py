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
