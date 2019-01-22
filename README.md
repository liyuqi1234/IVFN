# IVFuseNet: Fusion of Infrared and Visible Images for Depth Prediction
This repository includes our model code and dataset for depth prediction task.

This code works on Python 3 & TensorFlow 1.4 and the images in this dataset are of actual road scenes captured while unmanned vehicle driving.

If this code and dataset are provided for research purposes, please see License section below.

## Model
This model's structure looks just like the following illustration:
![Network](/pics/Fig2.png)

### Train
```python
python train2.py
```
### Test
```python
python test.py
```

## Data
Our NUST-SR dataset is composed of the actual road scenes captured while unmanned vehicle driving in the daytime and night. Currently the dataset contains visible light images, infrared images and depth map:

```
                        |daytime| night  
------------------------------------------
infrared images         | 6529  |  5612
visible light images    | 6529  |  5612
raw depth maps          | 6529  |  5612
------------------------------------------
```

The raw depth map should be preprocessed, firstly, filled the points for which there is no depth value first using colorization scheme of Levin et al in the NYUDepth development kit:

```python
python fill_points.py
```

Secondly, cut the visible light images, infrared images and depth map into 256*512:

```python
python rgb_ir_depth.py
```

Finally, shuffle images and generate the dataset:

```python
python shuffle picture.py
```
Snow_data_train and Snow_data_test are examples of train dataset and test dataset.

