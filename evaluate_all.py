from copy import deepcopy
import os
import subprocess

from evaluate import get_args_parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    root_dir = args.eval_model
    for subdir in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, subdir, 'monitoring_gpu.json')): # only if training finished
            args_copy = deepcopy(args)
            args_copy.eval_model = os.path.join(root_dir, subdir)
            run_args = ["python", "evaluate.py"]
            for key, val in args_copy.__dict__.items():
                if isinstance(val, bool):
                    if val:
                        run_args.append('--' + key.replace('_', '-'))
                else:
                    run_args.extend(['--' + key.replace('_', '-'), str(val)])
            subprocess.run(run_args)
