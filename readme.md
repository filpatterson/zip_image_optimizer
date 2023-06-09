First, install Miniconda latest. Then run following commands:

```conda create --name final_test_env python=3.9```

Consider that ```final_test_env``` is name of the environment that you create. You can set any name you want. After environment is created:

```conda activate final_test_env```

This will set you into created environment. Then install all dependencies using:

```python install -r requirements.txt```

Then run script using:

```python images_face_zip_compresser.py images.zip temporary compressed```

Here script will take ```images.zip``` file and take out of it all images into ```temporary``` folder. Then, all images will be compressed via cropping faces out of it and save them into ```compressed``` folder. Final step is that images from ```compressed``` will be taken to form new ```zip``` archive with name of original ```zip``` with added ```_compressed``` to the name. Both temporary folders and original ```zip``` will be removed.

In case you want to integrate it into bigger code, check ```Making Final Solution``` chapter of the ```.ipynb``` file.

### UPDATE:

Considering that there can be problems of mediapipe running on some computers (or you don't like mediapipe) there is a LBP classifier that performs great in terms of speed and efficiency. Therefore, here are added local models of the LBP classifier and Haar Classifier. The LBP classifier is used via next command:

```python opencv_images_face_zip_compresser.py images.zip temporary compressed```

If you want a Haar cascade one - check attached ```ipynb``` file