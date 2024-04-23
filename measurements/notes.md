### Notes on Integration

1) Lots of dependencies issues
- Created virtual environment `myenv` (see ReadMe to activate) for Python 3.8
- Make sure to have the right dependency versions. `pip3 install -r requirements.txt`
- Virtual environment requires `pip3` for installation and `python` to use Python3. I'm not sure why
- Remade a new `gmm_color_model_0_24_2.pkl` file for a different version of sklearn which wasn't compatible

2) Created a `utils` folder to hold models

3) I uploaded the following files, but I couldn't figure out how to test the end-to-end integration, so I may need a little bit of help running it properly to see if the updates were complete. 
    - `main.py`: Updated
    - `test_on_client.py`: Updated 
    - `test_on_client.py`: No changes (I wasn't sure what this was)
    - `sample_run.py`
        - Updated
        - Should we separate loading client and server models?
        - I have not put in boards and descriptions yet
