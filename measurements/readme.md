### Install dependencies

Install all libraries used via `pip3 install -r requirements.txt`. If you encounter permission issues, run the command with `sudo`.

If working on one of the 525 VMs, please execute `sudo yum install python3-devl` and `sudo yum install libjpeg-turbo-devel zlib-devel` first.


### Measure latencies

0. Source virtual environment `source myenv/bin/activate`

1. (Under `/measurements`) Start the api server with `python -W ignore -m uvicorn app.main:app --reload --host 0.0.0.0`

2. To measure the latency of our designed structure (creating embedding on client machine and pass to server for recommendation), execute the testing script with `python -W ignore test_on_client.py`, which prints `Time taken for reading image and calling api: {duration} seconds`. This excludes the time it takes to load ResNet.