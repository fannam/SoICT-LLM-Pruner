from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="soict-llm-pruner",
    version="0.1.0",
    description="A tri-level framework for structured pruning (Currently supports Llama, Qwen2, and Mistral)",
    author="Phan Hoang Nam",
    author_email="phanhoangnam234@gmail.com/nam.ph215434@sis.hust.edu.vn",
    packages=find_packages(),
    py_modules=["utils"],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.2",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "accelerate>=0.4.10",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
