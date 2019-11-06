from sacred import Experiment
import yaml
from pprint import pprint
import os
from sklearn.model_selection import ParameterGrid

ex = Experiment('grid_run')


def squish_dict(d):
    ks = list(d.keys())
    for k in ks:
        v = d[k]
        assert type(k) == str, 'cannot squish non-string keys'
        if type(v) != dict:
            continue

        squished = squish_dict(v)
        for sk, sv in squished.items():
            d[k+'.'+sk] = sv

        del d[k]
    return d

# # %%
# d = {
#     'a': 1,
#     'b': {
#         'c': 2,
#         'd': 3,
#         'flab': {
#             'foo': 67,
#             'bar': 'no'
#         }
#     },
#     'e': 'lol'
# }
# squish_dict(d)
# # %%


@ex.config
def config():
    cmd = 'echo'
    config_file = 'grid_config.yaml'
    same_seed = False


@ex.automain
def main(cmd, config_file, same_seed, seed):
    with open(config_file, 'r') as f:
        config = yaml.load(f.read())

    config = squish_dict(config)
    for k, v in config.items():
        if type(v) != list:
            config[k] = [v]

    extra = f" seed={seed}" if same_seed else ""
    pprint(config)

    param_grid = ParameterGrid(config)
    print(f"Parameter grid contains {len(param_grid)} combinations")
    print("Submitting jobs...")
    # param = param_grid[0]
    for param in param_grid:
        opts = " ".join([f"{k}={v}".replace(' ', '') for k, v in param.items()])
        torun = f"{cmd} with {opts}{extra}"
        print(torun)
        os.system(torun)
    print(f" Submitted {len(param_grid)} jobs")
