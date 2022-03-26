from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os
import matplotlib.pyplot as plt 

# Setting logging 


def checkUploadPath(path: str) -> bool:
    """Checks if the directory for uploads has been made or not

    Args:
        path (str): The file path which you would like to check

    Returns:
        bool: An indicator whether the path exists or not
    """
    if os.path.exists(path):
        return True
    else:
        return False


def createUploadPath(path: str) -> None:
    """Creates the folder for uploading images

    Args:
        path (str): The filepath where you would like to create the folder
    """
    os.makedirs(path)

def see_examples(generator, figsize=(16,16), nrows=4, ncols=4,random_state=42):
    """See random examples from the image data generator

    Args:
        generator (ImageDataGenerator): The generator from which you would like to see the examples
        figsize (tuple, optional): The size of matplotlib figure. Defaults to (16,16).
        nrows (int, optional): The number of rows in the figure. Defaults to 4.
        ncols (int, optional): The number of columns in the figure. Defaults to 4.
        random_state (int, optional): The numpy seed to be used while selecting random examples. Defaults to 42.
    """    
    # Initialize a label dictionary 
    label_dict = dict(zip(generator.class_indices.values(), generator.class_indices.keys()))
    # Calculate the number of batches in the data 
    num_batches = len(generator)
    # Set the seed for generating random numbers 
    np.random.seed(random_state)
    # Generate random batch indices 
    random_batch_indices = np.random.randint(low=0, high=num_batches, size=(nrows,ncols))
    # Generate the figure
    _, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    # Loop through the rows 
    for i in range(nrows):
        # Loop through the columns 
        for j in range(ncols):
            # Get the corresponding batch 
            images, labels = generator[random_batch_indices[i,j]]
            # Generate a random index for the batch examples 
            example_idx = np.random.randint(low=0, high=len(images))
            # Get the image at that index
            image = images[example_idx]
            # Get the label at that index
            label = labels[example_idx]
            # Get the label name from the label dictionary
            label = label_dict[np.where(label == 1)[0][0]]
            # Show the image 
            ax[i,j].axis('off')
            ax[i,j].imshow(image)
            ax[i,j].set_title(label)
    plt.tight_layout()
    plt.show()

def predictNew(model, path: str, labelDict=None, conf=True) -> tuple:
    """Gives prediction on an image

    Args:
        model (tf.keras.models.Model): The model which you would like to use for prediction
        path (str): The path where the image to be predicted is stored
        labelDict (dict, optional): The dictionary which contains the. Defaults to None.
        conf (bool, optional): Whether you would like to have the confidence intervals also. Defaults to True.

    Returns:
        tuple: A tuple containing confidence intervals and the predicion if conf=True, else it returns the prediction only
    """
    img = load_img(path, target_size=(256, 256))
    img = np.expand_dims(img, axis=0) # batchsize, height, width, rgb
    pred = model.predict(img) 
    predIdx = np.argmax(pred, axis=1)[0] 
    name = labelDict[predIdx]
    if conf:
        confidence = np.max(pred, axis=1)[0]
        return name, confidence
    else:
        return name