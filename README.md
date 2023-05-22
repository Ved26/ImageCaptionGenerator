# ImageCaptionGenerator

The project uses FLickr8k dataset that consists of 8000 images along with 5 captions per image. The dataset can be downloaded here: https://www.kaggle.com/datasets/waelboussbat/flickr8ksau/code<br>

For word-embeddings Glove-200d embeddings are used which can be downloaded here: https://www.kaggle.com/datasets/incorpes/glove6b200d
<br>

Please make sure that before to run the file, a folder "datasets" needs to be created with following structure:
<br>
  1. Flickr_Data(folder)
  2. flickr8k-sau (folder)
  3. test_imgs (folder)
  4. glove.6B.200d 
<br>

Install the required dependencies:
<br>

```pip install -r requirements.txt```

<br>
Run the  notebook "ImageCaptioning.ipynb" and save "model_30.h5"
<br>

<br>
Run app.py and the local web app will open.
<br>


```python app.py```

<br>
