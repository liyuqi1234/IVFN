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

## Data_Preprocess
Our NUST-SR dataset is composed of the actual road scenes captured while unmanned vehicle driving in the daytime and night. Currently the dataset contains visible light images, infrared images and depth map:

```
                        |daytime| night  
------------------------------------------
infrared images         | 6529  |  5612
visible light images    | 6529  |  5612
raw depth maps          | 6529  |  5612
------------------------------------------
```

The raw depth map should be preprocessed, firstly,  match the visible light images, infrared images and depth map, cut them into 256*512:

```python
python rgb_ir_depth.py
```

Secondly,filled the points for which there is no depth value first using colorization scheme of Levin et al in the NYUDepth development kit:

```python
python fill_points.py
```

Finally, shuffle images and generate the dataset （the final depth map are classified into 32 classes）:

```python
python shuffle picture.py
```
The example of raw depth map, depth map with filled points and the final depth map are shown below:
![Network](/pics/数据预处理示意图.png)

Here we give the data link , please click the link to get the visible light images, infrared images and the corresponding ground-truth depths.
[NUST-SR](https://pan.baidu.com/disk/home#/all?vmode=list&path=%2FNUST-SR)




### usage of dataset

Start with importing `package`:
```python
import h5py
import matplotlib.pyplot as plt
```
- To load a dataset:
```python
def read_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        images = np.asarray(f['images'])
        depths = np.asarray(f['depths'])
        infrareds = np.asarray(f['infrareds'])
    return images,depths,infrareds
images,depths,infrareds=read_hdf5('test_snow_data.h5')
```
- To display the image of a dataset:
```python
i, j = 0,4
imageTest = images[i:j]
plt.imshow(imageTest[0],cmap='jet')
```


## License
I provide this project for research purposes, please follow `Citing`.

For removal of copyrighted content, please contact me on GitHub.


## Citing
If you use this project in academic work, please cite as follows:

```bibtex
@misc{liyuqi1234,
        title={IVFuseNet: Fusion of Infrared and Visible Images for Depth Prediction},
        url={https://github.com/liyuqi1234/IVFN},
        author={Yuqi, Li},
        year={2019},
        publisher = {GitHub},
        journal = {GitHub repository}
}
```
