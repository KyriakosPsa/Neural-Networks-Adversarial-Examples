def create_CNN_model(input_shape=(28, 28, 1),
                     conv_layers=1,
                     head_layers=1,
                     activation='relu',
                     units=32,
                     batchnorm=False,
                     maxpooling=False,
                     maxpooling_every_n_cov_layer=1,
                     dropout=False,
                     dropout_rate=0.1,
                     data_augmentation=False,
                     filters=32,
                     node_increase_per_layer=0,
                     filters_increase_per_layer=0):

    # Model architecture
    model = keras.Sequential(name="MNIST_classifier_CNN_Model")
    model.add(layers.InputLayer(input_shape=input_shape, name='Input'))

    # Apply shifting to the images per batch
    if data_augmentation:
        model.add(layers.RandomTranslation(height_factor=0.15,
                                           width_factor=0.15,
                                           fill_value=0.0, name="Random_shift"))

    # Number Convolution layer(s) (parameter) ----------------------------------------
    # Initialize Convolution count
    conv_layer_count = 0
    # Initialize increase_filters_by
    increase_filters_by = 0
    for convolutions in range(conv_layers):
        # Add a Convolutional layer
        model.add(layers.Conv2D(filters=filters + increase_filters_by, kernel_size=(5, 5), padding='same',
                                activation=activation, name=f"Convolution_number_{convolutions+1}"))
    # If Batch normalization is True, add a Batch normalization layer
        if batchnorm:
            model.add(layers.BatchNormalization(
                name=f"Batch_Normalization_{convolutions+1}"))
        # If dropout is True, add a Dropout layer with a parameter rate (parameter)
        if dropout:
            model.add(layers.Dropout(
                dropout_rate, name=f"Dropout_layer{convolutions+1}_rate{dropout_rate}"))
        # If Maxpooling is True, add a maxpooling layer
        conv_layer_count += 1
        if (maxpooling == True) and (conv_layer_count == maxpooling_every_n_cov_layer):
            model.add(layers.MaxPool2D(pool_size=(2, 2),
                      name=f"MaxPool_2by2_{convolutions+1}"))
            # Set the counter back to zero to be ready for the next maxpooling layer
            conv_layer_count = 0
        # Create the next convolutional layer with this many less filters
        increase_filters_by += filters_increase_per_layer

    # Number of MLP head layers(s) (parameter) ----------------------------------
    # Initialize increase_units_by
    increasee_units_by = 0
    model.add(layers.Flatten(name="Flatten_layer"))
    for head in range(head_layers):
        # Add a Dense layers
        model.add(layers.Dense(units=(units + increasee_units_by),
                  activation=activation, name=f"FFN_Classifier_layer_{head}"))
        # If Batch normalization is True, add a Batch normalization layer (parameter)
        if batchnorm:
            model.add(layers.BatchNormalization(
                name=f"Batch_Normalization_{conv_layers+1+head}"))
        # If dropout is True, add a Dropout layer with a parameter rate (parameter)
        if dropout:
            model.add(layers.Dropout(
                dropout_rate, name=f"Dropout_layer{conv_layers+1+head}_rate{dropout_rate}"))
        # Create the next feed forward layrs with this many less nodes
        increasee_units_by += node_increase_per_layer

    # Output layer --------------------------------------------
    model.add(layers.Dense(10, activation='softmax', name="Output_Layer"))

    # Model optimizer and compiling (static)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model
