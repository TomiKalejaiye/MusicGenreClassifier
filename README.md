# MusicGenreClassifier

![Main Image](https://github.com/TomiKalejaiye/MusicGenreClassifier/blob/master/images/main_img.png)

A music genre classification system that uses a convolutional neural network to classify songs into one of 10 genres based on the GTZAN dataset. MATLAB is used to take short time discrete Fourier transforms (ST-DFTs) of raw audio files. In Python these are then converted to mel-spectrograms, which are used to train a convolutional neural network (implemented with PyTorch) to predict the genre of the song from a set of 10 genres. The trained network achieves a classification accuracy of 82% on the randomly split test set.
