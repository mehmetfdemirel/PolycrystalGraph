# Polygrain Microstructure Property Prediction Project w/ [Mesoscale Computational Modeling Research Group](https://mesomod.weebly.com/datasets.html)

This repo contains the microstructure-property dataset and code base for the graph neural network-based polygrain microstructure property prediction project.

## Microstructure-property dataset for polycrystalline materials
We use [Dream.3D](http://dream3d.bluequartz.net/) to generate 492 different 3D polycrystalline microstructures. The number of grains in each microstructure varies from 12 to 297 grains. Microstructures with and without strong textures are both generated. For each microstructure, we performed phase-field modeling to obtain the 3D distributions of local magnetization and the associated local magnetostriction induced by a magnetic field applied along the x-axis. Four or five different magnetic fields are applied to each microstructure, amounting to 2287 data points.

## Run machine learning code on server

### 1. Setting up environment
```
conda env create -f env.yml
conda activate matscienv
```

### 2. Running the code

#### 2.1. Data Split
```
python split_validation.py
```

#### 2.2. Training
Either run  
```
bash run.sh
```
or  
```
python modelcv.py --epoch=<num_of_epochs> \
		  --learning_rate=<learning_rate> \
		  --batch_size=<batch_size> \
		  --latent_dim=<latent dimension between the two layers of the GNN> \
		  --max_node_num=<maximum number of nodes in a graph in the entire dataset>
```
## Run machine learning code on Google Colab
### 1. open the notebook GNN_interpretation.ipynb

