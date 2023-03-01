# GRLSTM, AAAI 2023

1. 
'''sh 
cd utils
'''

python construct_KG.py

move files 'dataset'_e_map.txt, 'dataset'_KG_graph.txt into 'KGE/datasets/beijing(newyork)/kg'
here are files which has been addressed.

2. cd KGE
python run_knowledge_representation.py

3.cd utils
python construct_fusion_graph.py

4.python Train.py

5.python Validation.py

6.python Test.py