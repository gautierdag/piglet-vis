# Learning the Effects of Physical Actions in a Multi-modal Environment (EACL-2023)

### Dataset

Our dataset split along with PIGPeN images are unfortunately not yet publicly available as they are several GBs and require a external hosting solution.
The dataset will be made publicly available soon.

### Setup

Please install dependencies in `requirements.txt`.

We use `wandb` for logging and so you will need to set that up separately to make use of wandb.

We also use `h5py` to store intermediate representations from frozen models (DETR and LLM) and depending on your OS you might need to install separate dependencies to ensure that `h5py` can run.

### Train

We use Hydra `conf` files to declare a run.

To reproduce the original Piglet model (Zellers et al., 2021):

`python code/main.py --config-name piglet`

To train our `base` baseline:

`python code/main.py --config-name base`

To train our `base+symbolic` model (smaller Piglet model):

`python code/main.py --config-name base_symbolic`

To train our `base+symbolic+images` model:

`python code/main.py --config-name base_symbolic_images`

To train our `base+image` model:

`python code/main.py --config-name base_images`

To train our `base+image-text-labels` model:

`python code/main.py --config-name base_images_text_labels`

If you wish to modify or vary other parameters such as `seed` you can do the following:

`python code/main.py --config-name base_images ++model.hidden_size=64 ++model.num_layers=3 ++pretrain.batch_size=256 ++seed=10`

### Evaluation

Evaluation and analysis is handled through the `wandb` logging.
