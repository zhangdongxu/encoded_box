# encoded_box

This is the code base for exploring neural encoders that takes node features as input and output boxes. We use hypernymy discovery task as testbed. 

0. Create a virtual enviroment:

`conda create -n <your_env_name> python==3.7`

When going through the following steps, you may install missing libraries via `pip install`

1. Create dataset and input features for hypernymy discovery:

`python src/generate_wordnet.py`

2. Training

`python src/main.py --help`

To do grid search: 

`python grid_search.py`

3. Collect results:

`python src/result_analysis.py`

