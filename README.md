```text
mylib/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── download_data.py
│
├── loggers/
│   ├── __init__.py
│   └── logger.py
│
├── models/
│   ├── __init__.py
│   ├── cnn/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── callbacks/
│   │   │   ├── __init__.py
│   │   │   ├── discriminative_lr.py
│   │   │   └── progressive_unfreeze.py
│   │   └── model/
│   │       ├── __init__.py
│   │       └── base_model.py
│   │
│   └── transformers/
│       ├── __init__.py
│       ├── config.py
│       ├── callbacks/
│       │   └── __init__.py
│       └── model/
│           ├── __init__.py
│           └── base_model.py
│
├── utils/
│   ├── __init__.py
│   └── visualization.py
│
├── config.py
├── README.md
├── requirements.txt
└── setup.py
```
