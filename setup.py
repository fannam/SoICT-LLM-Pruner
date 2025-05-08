from setuptools import setup, find_packages
from pathlib import Path

# long description from README.md if available
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="soict-llm-pruner",
    version="0.1.0",
    description="Enhancing LLM Performance via Structured Pruning and Knowledge Distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="fannam",
    url="https://github.com/fannam/SoICT-LLM-Pruner",
    license="MIT",
    packages=find_packages(include=[
        "block_level_pruner",
        "element_level_pruner",
        "layer_level_pruner",
        "estimator",
        "teacher_correction",
        "utils",
        # if your code is placed under a top-level package, use that instead:
        # "soict_llm_pruner",
    ]),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "tqdm>=4.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
