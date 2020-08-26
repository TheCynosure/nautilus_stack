## Learning Loop Closure
Using embeddings of point clouds to determine good locations for loop closure.

This repository trains a neural net based loosely on PointNet (Qi et. al, 2016) to create rotation and translation-invariant embeddings of Point Clouds from laser scans.

We train this network using Triplet Loss, where each cloud's embedding is expected to be similar to those that are from a nearby location, and different from those at some random other location.

The `learning` folder contains the necessary logic for training.
`dataset.py` handles extracting training data from source files and creating an `LCDataset` object from them
`model.py` defines the network itself.
`trian.py` handles the training of the network.
`embed.py` can take a pretrained network and output embeddings.

### Installing

You will need some dependencies:
```
pip3 install torch scipy
```

#### Training
To train the embedder network, simply run
```
python train.py --dataset ../data/dataset_name
```
This will train with the default parameters, saving completed models' state dicts in the `cls` folder.

See
```
python train.py --help
```
For additional options.

#### Creating Embeddings
To get the embeddings for a particular set of point clouds using a particular pretrained PyTorch model, run:
```
python learning/embed.py --model [model_path] --data_path [data_path] --out_path [out_path]
```
if `data_path` points to a directory, we will create a directory of corresponding `.embedding` files at `out_path`

#### Data
The `data` folder contains training data, which should be of the following structure:

```
data/
    dataset_name/
        point_*timestamp*.data
        ...
        point_*timestamp*.data.location
        ...
	dataset_info.json
```
Here, each .data file contains a bunch of newline-separated point locations of the form "x y z", and each .location file contains the corresponding "location" of the form "x y z".
The json file contains general information like dataset name, time, as well as divisions of the data into training, dev, and evaluation sets
