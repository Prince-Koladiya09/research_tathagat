from setuptools import setup, find_packages

setup(
    name="adv_image_finetune",
    version="0.2.0",
    description="An advanced library for fine-tuning image classification models.",
    author="Meet Vyas, Prince Koladiya",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12",
        "keras>=2.12",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "kagglehub",
        "pillow",
        "pydantic",
        "tensorflow_hub",
        "joblib"
    ],
    extras_require={
        "viz": [
            "opencv-python>=4.8.0",
            "umap-learn>=0.5.4",
            "plotly>=5.16.0",
            "lime",
            "shap",
        ]
    },
    python_requires=">=3.8",
)