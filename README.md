# Visual Misinformation

### Software structure 

```
.
├── data
│   ├── .hf.txt (HuggingFace API key)
│   ├── .key.txt (OpenAI API key)
│   ├── dialogs.json
│   └── news.json
├── imgs (contains .jpg files)
├── res (contains folders of results)
└── src
    ├── complete.py
    ├── load_cnfg.py
    ├── main_exec.py
    ├── models.py
    ├── prompt.py
    ├── save_res.py
    └── cfg_###.py (any config file)
```

To run the program, navigate to the `src` directory and execute a command like the following:
```
$ python main_exec.py -c cfg_example
```
