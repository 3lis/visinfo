# Visual Misinformation

### Software structure 

```
.
├── data
│   ├── .hf.txt (HuggingFace API key)
│   ├── .key.txt (OpenAI API key)
│   ├── dialogs.json
│   └── news.json
├── imgs (contains JPG files)
├── res (results are saved here)
└── src
    ├── complete.py
    ├── conversation.py
    ├── load_cnfg.py
    ├── main_exec.py
    ├── models.py
    ├── prompt.py
    ├── save_res.py
    └── cfg_###.py (any config file)
```
Examples of `imgs` and `res` are on [Google Drive](https://drive.google.com/drive/folders/13mso0QZPu3A9fsY5-anVy0xuWAUOVcBu).

To install the requirements, on the root folder of the project execute the following command:
```
$ pip install -r req.txt
```

To run the program, navigate to the `src` directory and execute a command like the following:
```
$ python main_exec.py -c cfg_example
```