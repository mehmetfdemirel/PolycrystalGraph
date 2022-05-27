# Graph Neural Networks for an Accurate and Interpretable Prediction of the Properties of Polycrystalline Materials 

This repo contains the code base for the paper [***"Graph Neural Networks for an Accurate
and Interpretable Prediction of the Properties of Polycrystalline Materials"***](https://www.nature.com/articles/s41524-021-00574-w)
by [Minyi Dai](https://www.linkedin.com/in/minyi-dai-7bb82b197/), 
[Mehmet F. Demirel](http://cs.wisc.edu/~demirel), 
[Yingyu Liang](http://cs.wisc.edu/~yliang), 
[Jiamian Hu](https://mesomod.weebly.com/people.html).

## Code and paper correction
We are aware that the message passing between neighboring nodes was not implemented in the layer-wise update function due to an error in the original code of the graph neural network (GNN) model. Thus, we update the code model.py and the optimized hyperparameters in this GitHub page. The changes to the original paper can be found in the [author correction](https://www.nature.com/articles/s41524-022-00804-9).

## Microstructure-property dataset for polycrystalline materials
We use [Dream.3D](http://dream3d.bluequartz.net/) 
to generate 492 different 3D polycrystalline microstructures. 
The number of grains in each microstructure varies from 12 to 297 grains. 
Microstructures with and without strong textures are both generated (see examples below). 
For each microstructure, we performed phase-field modeling
to obtain the 3D distributions of local magnetization and the associated
local magnetostriction induced by a magnetic field applied along the x-axis.
Four or five different magnetic fields are applied to each microstructure,
amounting to 2287 data points.
![alt text](https://github.com/mehmetfdemirel/microstructure/blob/master/img/microstructure.png)

## Run machine learning code

### 1. Set up Conda environment
```
conda env create -f env.yml
conda activate micstrenv
```

### 2. Download the data

```
bash download_data.sh
```

### 3. Run the code

#### 3.1. Split data for cross validation
```
python split.py
```

#### 3.2. Train the model
Run  
```
bash run.sh
```
##### 3.3. Get Interpretation results
```
python interpretation.py
```



