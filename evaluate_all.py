from copy import deepcopy
import os
import subprocess

from evaluate import get_args_parser


def create_subcommand(args):
    run_args = ["python", "evaluate.py"]
    for key, val in args_copy.__dict__.items():
        if isinstance(val, bool):
            if val:
                run_args.append('--' + key.replace('_', '-'))
        else:
            run_args.extend(['--' + key.replace('_', '-'), str(val)])
    return run_args


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--timeout", default=3600, type=int, help="timeout for each evaluate subcall")
    args = parser.parse_args()
    timeout = args.timeout
    delattr(args, 'timeout')
    if os.path.isdir(args.eval_model):
        # run evaluation for every subdir
        root_dir = args.eval_model
        for subdir in os.listdir(root_dir):
            if os.path.isfile(os.path.join(root_dir, subdir, 'monitoring_gpu.json')): # only if training finished
                args_copy = deepcopy(args)
                args_copy.eval_model = os.path.join(root_dir, subdir)
                run_args = create_subcommand(args_copy)
                try:
                    subprocess.run(run_args, timeout=timeout)
                except Exception as e:
                    print(e)

    else:
        # comma-separated list of models given, run eval on pretrained versions
        models = args.eval_model.split(',')
        for model in models:
            args_copy = deepcopy(args)
            args_copy.eval_model = model
            run_args = create_subcommand(args_copy)
            try:
                subprocess.run(run_args, timeout=timeout) # TODO might need to increase this later
            except Exception as e:
                print(e)
