# Face Recognition:
  Identify a face using deep learning model.
  
``` Our model will use pretrained VGG-16 model trained on ImageNet dataset

It has total 23 layers
1 InputLayer        6 Conv2D          11 MaxPooling2D     16 Conv2D         21 Dense
2 Conv2D            7 MaxPooling2D    12 Conv2D           17 Conv2D         22 Dense
3 Conv2D            8 Conv2D          13 Conv2D           18 Conv2D         23 Dense  **Output_Layer**
4 MaxPooling2D      9 Conv2D          14 Conv2D           19 MaxPooling2D
5 Conv2D            10 Conv2D         15 MaxPooling2D     20 Flatten
```

Instead of using same network we will **Fine Tune** it for our task.

> I know that VGG-16 isn't best for this task. We can use FaceNet or OpenFace for it but just for sack of demonstration, I'm gonna use VGG-16

Step 1. Finalize the architecture we'll gonna use.
        We'll first remove last four layers so that we can fine tune it.
        now our acrhitecture is look like this
        
```
  1 InputLayer        6 Conv2D          11 MaxPooling2D     16 Conv2D        
  2 Conv2D            7 MaxPooling2D    12 Conv2D           17 Conv2D       
  3 Conv2D            8 Conv2D          13 Conv2D           18 Conv2D        
  4 MaxPooling2D      9 Conv2D          14 Conv2D           19 MaxPooling2D
  5 Conv2D            10 Conv2D         15 MaxPooling2D     
```
        
        
        
  Code for this 
```python
  from keras.applications import VGG16
  model = VGG16(weights = 'imagenet',       # download weights for ImageNet
              include_top = False,          # Remove the last 4 layers
              input_shape = (224,224,3) )   # VGG-16 expect input of size (224 , 224 , 3)
```
Step 2. Now freeze the all remaining layers because we don't want to train the model from scratch.

> You can refer this concept as **Transfer Learning**.

```python
  for layer in model.layers:      # iterate over all layers
  layer.trainable = False         # set trainable as False

  # Print all layers  
  for (i, layer) in enumerate(model.layers):
  print(str(i)+ " " + layer.__class__.__name__, layer.trainable)
```

Output should be look like this

```
0 InputLayer False
1 Conv2D False
2 Conv2D False
3 MaxPooling2D False
4 Conv2D False
5 Conv2D False
6 MaxPooling2D False
7 Conv2D False
8 Conv2D False
9 Conv2D False
10 MaxPooling2D False
11 Conv2D False
12 Conv2D False
13 Conv2D False
14 MaxPooling2D False
15 Conv2D False
16 Conv2D False
17 Conv2D False
18 MaxPooling2D False
```
Now we've our model ready for Fine Tuned.

Step 3. Modify the architecture by adding some more layers
```python
def addTopLayers(bottom_layer , num_classes , units = 256 ):
    top_model = bottom_layer.output
    top_model = Flatten(name = "flatten" )(top_model)
    top_model = Dense(units , activation = "relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(num_classes , activation = "softmax")(top_model)
    return top_model
```


```python
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Activation , Flatten , Conv2D , MaxPooling2D , ZeroPadding2D
from keras.layers.normalization import BatchNormalization

new_model = addTopLayers(model , 5 )
new_model = Model(input= model.input , output = new_model)
new_model.summary()
```

Step 4. Prepare the data 

So I've included the Dataset in repo. You can find more information [here](../master/facerecog/dataset)

you can download it by downloading the repository

```
git clone https://github.com/rajneeshmehta/facerecog.git
```
Now now we've the complete dataset we can start working. 
We've total 3836 images of total five classes.
We'll split it into 80% - 20% train and validation data.

```python
from keras.preprocessing.image import ImageDataGenerator                      # load ImageDataGenerator 
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)     #create a data generator and specifiy validation split size 

train_dataset = image_generator.flow_from_directory(batch_size=32,                        # load 32 image at a time
                                                 directory='/content/facerecog/dataset',  # specify the dataset directory
                                                 shuffle=True,                            # randomize the images
                                                 target_size=(224, 224),                  # remember Vgg-16 requires 224 X 224 images
                                                 subset="training",     
                                                 class_mode='categorical')                # since we've multiple classes 

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='/content/facerecog/dataset',
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="validation",
                                                 class_mode='categorical')
```
Now our data is ready to use.

Step 5. 
https://gist.github.com/rajneeshmehta
