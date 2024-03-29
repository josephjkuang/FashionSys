### Install dependencies

Install all libraries used via `pip install -r requirements.txt`. If you encounter permission issues, run the command with `sudo`.

If working on one of the 525 VMs, please execute `sudo yum install python3-devl` and `sudo yum install libjpeg-turbo-devel zlib-devel` first.


### Run the sample script

1. Download the [Fashion Product Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data)

2. Update `sample_run.py` so that `in_path = <PATH_TO_FASHION_PRODUCET_DATASET>` and `out_path = <PATH_FOR_STORING_RESULTS>`

3. Execute `sample_run.py` with `python3 sample_run.py`