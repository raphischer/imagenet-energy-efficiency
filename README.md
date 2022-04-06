# Assessing Energy Efficiency of Machine Learning - MLEE

Code and results for assesing energy efficiency of machine learning tasks.
The associated research paper is currently under review, upon acceptance this repository will be published on the author's GitHub.

## Installation
All code was executed with Python 3.8, please refer to [requirements](./requirements.txt) for all dependencies.
Depending on how you intend to use this software, only some packages are required.

## Usage
To investigate the results you can use our publicly available [Energy Label Exploration tool](http://167.99.254.41/), so no code needs to be run on your machine.

To start [ELEx](elex.py) locally, simply call `python elex.py` and open the given URL in any webbrowser.
Call `python -m mlee.label_generator` to generate an energy label either for a given task / model / environment, or any of the merged logs (provided via command line).
The [results](./paper_results/) (plots and tables) in the paper were generated with the corresponding [script](create_paper_results.py).

New experiments can also be executed, available tasks are [inference](infer.py) and [training](train.py).
You can pass the chosen model, software backend and more configuration options via command line.
For `--data-path` pass the directory with full `ImageNet` data for the chose software `--backend`, refer to the respective implementations for [TensorFlow](./mlee/ml_tensorflow/load_imagenet.py) and [PyTorch](./mlee/ml_pytorch/train.py).
For each experiment a folder is created, which can be [merged](merge_results.py) into more compact `.json` format.
Note that due to monitoring of power draw, we mainly tested on limited hardware architectures and systems (Linux systems with Intel CPUs and NVIDIA GPUs).
You can also inspect the [scripts](./scripts/) used to run all esxperiments.

## Road Ahead
We intend to extend and improve our software framework:
- polish the ELEx tool, allow to execute expeirments locally from GUI
- support more implementations, monitoring options, models, metrics, and tasks.
- move beyond sustainability and incorporate other important aspects of trustworthiness
- more improvements based on reviewer feedback

## Reference & Term of Use
Please refer to the [license](.LICENSE.md) for terms of use.
If you use this code or the data, please link back to https://github.com/tmplxz/ml-energy-efficiency.

Copyright (c) 2022 tmplxz
