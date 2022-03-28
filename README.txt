# Real time Emotion Detection with convolutional neural networks

<p align="center">
  <img width="800" src="emotions.gif">
  
</p>


Install python and pip, then run:

pip install -r requirements.txt

After that, simply execute any of the files you'd like to use.

If you'd like to train the model from strach on the FER dataset,
simply download it from kaggle and put it in the same folder as these scripts before running the train file.
https://www.kaggle.com/datasets/msambare/fer2013

Google colab can be used to run the jupyter notebook or python files online for free with GPUs and TPUs if your machine is not powerful enough to run
the training algorithm.
https://colab.research.google.com/

It is also possible to simply visualize the notebook online in this link:
https://nbviewer.org/

To execute python files on the terminal, simply run:

python test.py

for example.

The train, test and predict files can either be executed by themselves or called as functions if imported as modules.

The exploratory data analysis jupyter notebook explains everything in simple terms, while the other files contain information on how to use them properly.

The train file trains a model from scratch, given a dataset of images.

The test file opens your webcam feed and shows the results in real time on your screen, until you hit Q to quit.

The predict file receives an image with a face as input and returns an emotion as output.
