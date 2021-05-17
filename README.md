# Convolutional Neural Network to Classify types of Skin Cancers

This was a pair-programming excersice, the team members are Tommaso Capecchi and myself. 

This project entails the implementation of a Convolutional Neural Network (CNN) which aims to identify and classify different types of skin cancers. This project uses the HAM10000 dataset, which contains dermatoscopic images. This dataset was artificially augmented to improve the system's accuracy and performance. 
The project is implemented in Python, and the frameworks used are Pytorch, Torchvision, Numpy and Matplotlib.



The Augmented_HAM10000_dataset file contains a class which enherits Pytorch's Dataset class to load the custom dataset. 

The augmented_data_2 is a csv file which contains the image names and target values of the original and augmented images (46890 in total)

The cnn_introtoAI file contains the CNN model 

The cnn_model_final file is the saved cnn model, trained on 500 epochs. 

