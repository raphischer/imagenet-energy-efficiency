# Assessing Energy Efficiency of Machine Learning - MLEE

Code and results for assesing energy efficiency of machine learning tasks.
The associated research paper is currently under review, upon acceptance this repository will be published on the author's GitHub.

## Installation
All code was executed with Python 3.8, please refer to [requirements](./requirements.txt) for all dependencies (not all packages are necessary, we listed packages for a range of environments).

## Usage
This repository contains all [experiment logs](./results/), which can be investigated with the interactive [Energy Label Exploration tool](elex.py) (based on `Dash`).
Simply call `python elex.py` and open the given (local) URL in any webbrowser.
The [results](./paper_results/) in the paper were generated with the corresponding [script](create_paper_results.py).
Call `python -m mlel.label_generator` to generate an energy label for any of the merged logs (e.g, pass `-f results/infer_2022_03_05_23_09_30.json`).

New experiments can also be executed, available tasks are [inference](infer.py) and [training](train.py).
You can pass the chosen model, software backend and more configuration options via command-line.
For each experiment a folder is created, which can be [merged](merge_results.py) into more compact `.json` format.
Note that due to monitoring of power draw, we mainly tested on limited hardware architectures and systems (Linux systems with Intel CPUs and NVIDIA GPUs).

## Road Ahead
We intend to extend and improve our software framework:
- improve the ELEx tool and make results investigatable from public URL
- support more implementations, monitoring options, models, metrics and tasks.
- move beyong sustainability and incorporate other important aspects of trustworthiness

## Reference & Term of Use
Please refer to the [license](.LICENSE.md) for terms of use.
If you use this code or the data, please link back to https://github.com/tmplxz/ml-energy-efficiency.

Copyright (c) 2022 tmplxz
