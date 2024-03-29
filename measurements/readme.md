### Install dependencies

Install all libraries used via `pip install -r requirements.txt`. If you encounter permission issues, run the command with `sudo`.

If working on one of the 525 VMs, please execute `sudo yum install python3-devl` and `sudo yum install libjpeg-turbo-devel zlib-devel` first.


### Measure latencies

1. Download the [Fashion Product Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data)

2. Update `sample_run.py` so that `in_path = <PATH_TO_FASHION_PRODUCET_DATASET>` and `out_path = <PATH_FOR_STORING_RESULTS>`

3. (Under `/measurements`) Start the api server with python3 -W ignore -m uvicorn app.main:app --reload

4. To measure the latency of running everything (creating embedding and giving recommendation) on server, use the command `time curl -X POST "http://127.0.0.1:8000/full_prediction/" -F "file=@<PATH_TO_INPUT_IMAGE>"`

5. To measure the latency of our designed structure (creating embedding on client machine and pass to server for recommendation), execute the testing script with `python3 -W ignore test_on_client.py`, which prints `Time taken for reading image and calling api: {duration} seconds`. This excludes the time it takes to load ResNet.