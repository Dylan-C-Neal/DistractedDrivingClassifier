def sampleCV(model,
             data
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