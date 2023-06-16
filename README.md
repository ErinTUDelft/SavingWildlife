# SavingWildlife
Welcome to the project in which we aim to provide automatic animal localization in order to enhance wildlife monitoring efforts. 

## Tree Setup:
- The architecture branch is the most up to date, and is recommended for use
- The Training and lightweight net were primarily made for prototyping
- The inference branch consists of only a tiny amount of data, and should be the one used for embedding on edge devices. If space is an issue it is recommended to just 
clone this specific branch, and use depth 1 to get only the latest version

## Architecture explanation
The most important file is the train_class_local, in which the training parameters can be defined. The create_dataloader file attaches the correct labels to the images, splits 
the images, and creates a train and test dataloader. The final state of the final model was saved every 5 epochs, and can be used for analysis, which is done in the test_class_local file
Support for investigating the bounding boxes is provided in val_class_local, and additional utility functions are defined in utils.py
