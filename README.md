# Polygrain Microstructure Property Prediction Project w/ [Mesoscale Computational Modeling Research Group](https://mesomod.weebly.com/datasets.html)

This repo contains the code base for the graph neural network-based polygrain microstructure property prediction project.

## 1. Setting up environment
```
conda env create -f env.yml
conda activate matscienv
```

## 2. Running the code

### 2.1. Data Split
```
python split_validation.py
```

### 2.2. Training
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

