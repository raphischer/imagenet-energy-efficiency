# Assessing Energy Efficiency of Machine Learning

Code and results for assesing energy efficiency of various ImageNet models, with an associated research paper published at [ECML PKDD 2022](https://link.springer.com/chapter/10.1007/978-3-031-23618-1_3).

## Big News
Please also check out our open access follow up work on [Sustainable and Trustworthy Reporting (STREP)](https://link.springer.com/article/10.1007/s10618-024-01020-3). It is accompanied by a public website (extension of the original ELEx tool) that allows to [investigate our experimental results](https://strep.onrender.com/?database=ImageNetEff22) without running any code locally! Accordingly, this code base will not receive any further updates, but please feel free to look into the [STREP software library](https://github.com/raphischer/strep) which will receive further updates.

## Installation
All code was executed with Python 3.8, please refer to [requirements](./requirements.txt) for all dependencies.
Depending on how you intend to use this software, only some packages are required.

## Usage
To start [ELEx](elex.py) locally, simply call `python elex.py` and open the given URL in any webbrowser.
Call `python -m mlee.label_generator` to generate an energy label either for a given task / model / environment, or any of the merged logs (provided via command line).
The [results](./paper_results/) (plots and tables) in the paper were generated with the corresponding [script](create_paper_results.py).

New experiments can also be executed, available tasks are [inference](infer.py) and [training](train.py).
You can pass the chosen model, software backend and more configuration options via command line.
For `--data-path` pass the directory with full `ImageNet` data for the chose software `--backend`, refer to the respective implementations for [TensorFlow](./mlee/ml_tensorflow/load_imagenet.py) and [PyTorch](./mlee/ml_pytorch/train.py).
For each experiment a folder is created, which can be [merged](merge_results.py) into more compact `.json` format.
Note that due to monitoring of power draw, we mainly tested on limited hardware architectures and systems (Linux systems with Intel CPUs and NVIDIA GPUs).
You can also inspect the [scripts](./scripts/) used to run all esxperiments.

## Reference & Term of Use
Please refer to the [license](.LICENSE.md) for terms of use.
If you appreciate our work and code, please cite [our paper](https://link.springer.com/chapter/10.1007/978-3-031-23618-1_3) as given by Springer:

Fischer, R., Jakobs, M., Mücke, S., Morik, K. (2023). A Unified Framework for Assessing Energy Efficiency of Machine Learning. In: Machine Learning and Principles and Practice of Knowledge Discovery in Databases. ECML PKDD 2022. Communications in Computer and Information Science, vol 1752. Springer, Cham. https://doi.org/10.1007/978-3-031-23618-1_3

or using the bibkey below:

```
@inproceedings{fischer_unified_2022,
	location = {Cham},
	title = {A Unified Framework for Assessing Energy Efficiency of Machine Learning},
	rights = {All rights reserved},
	doi = {10.1007/978-3-031-23618-1_3},
	pages = {39--54},
	booktitle = {Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
	publisher = {Springer Nature Switzerland},
	author = {Fischer, Raphael and Jakobs, Matthias and Mücke, Sascha and Morik, Katharina},
	date = {2022},
}
```

Copyright (c) 2022 Raphael Fischer
