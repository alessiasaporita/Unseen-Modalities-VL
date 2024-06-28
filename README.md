## [Reproducing Unseen Modality Interaction for vision and language classification](https://arxiv.org/abs/2306.12795) (NeurIPS 2023)


<img width="1595" alt="Screenshot 2024-01-22 at 16 38 34" src="https://github.com/xiaobai1217/Unseen-Modality-Interaction/assets/22721775/ecc432fb-722d-41bc-befc-4add1a5abb5d">

This is the code for the vision and language classification task using MM-IMDb, UPMC Food-101, and Hateful Memes with image and text modalities. We apply to vision and language domain the framework proposed by Yunhua Zhang, Hazel Doughty, Cees G.M. Snoek Learning Unseen Modality Interaction In NeurIPS, 2023. 


### Environment
* Python 3.8.5
* torch 1.9.0+cu111
* torchaudio 0.9.0
* torchvision 0.10.0+cu111

### Dataset
We use three vision and language datasets: [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb), [UPMC Food-101](https://visiir.isir.upmc.fr/explore), and [Hateful Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/). We use `pyarrow` to serialize the datasets, the conversion codes are located in `vilt/utils/write_*.py`. Please see `DATA.md` to organize the datasets, otherwise you may need to revise the `write_*.py` files to meet your dataset path and files. Run the following script to create the pyarrow binary file:
```
python make_arrow.py --dataset [DATASET] --root [YOUR_DATASET_ROOT]
```

### Web dataset
We create the webdatset by `webdatset.py`

### Run Demo
* We provide the splits for training, validation and testing in the `annotations` folder. 

* Commands for all experiments / models can be found in the file `bash2.sh`.


