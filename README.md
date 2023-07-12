# [AAAI 2023] GRLSTM: Trajectory Similarity Computation with Graph-based Residual LSTM

This is a PyTorch implementation of GRLSTM, as described in our paper: Silin Zhou, Jing Li, Hao Wang, Shuo Shang, and Peng Han.

## Framework
![Framework](framework.png#pic_center)

## Experiment
![Exp](exp.png#pic_center) 

## If you want to reproduce the results, please follow the steps below

### construct knowledge graph
```shell
cd utils
python construct_KG.py
```

### move, rename, create files
1. move file ***bj(or ny)_e_map.txt*** into ***KGE/datasets/beijing(or newyork)/kg*** and rename ***e_map.txt***

2. move file ***bj(or ny)_KG_graph.txt*** into ***KGE/datasets/beijing(or newyork)/kg*** and rename ***train.txt***

3. create a empty file and name ***test.txt***

4. create a file named ***r_map.txt*** and copy the follow text
```shell
road 0
traj_in 1
traj_not_in 2
```

### train knowledge graph embedding
```shell
cd KGE
python run_knowledge_representation.py
```

### construct fusion graph
```shell
cd utils
python construct_fusion_graph.py
```

## Train
```shell
python Train.py
```

## Validation
```shell
python Validation.py
```

## Test
```shell
python Test.py
```
