# utils/loss_functions.py
import torch
import numpy as np
import random
import copy
from diskcache import Index
from fl_testing.frameworks.models import get_pytorch_model, sum_model_weights_pytorch

def seed_every_thing(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fedavg_aggregate(models_state_dict, num_samples):
    # Ensure the list of models and number of samples have the same length
    assert len(models_state_dict) == len(
        num_samples), "The number of models must match the number of sample counts"

    # Initialize a model with the same architecture as the client models
    global_model_state_dict = copy.deepcopy(models_state_dict[0])

    # Initialize a dictionary to store the weighted sum of parameters
    global_state_dict = {key: torch.zeros_like(
        value) for key, value in global_model_state_dict.items()}

    # Total number of samples across all clients
    total_samples = sum(num_samples)

    # Perform weighted aggregation of the client models
    for state_dict, n in zip(models_state_dict, num_samples):
        # Update global model parameters with the weighted sum
        for key in global_state_dict.keys():
            val = state_dict[key]
            if val.is_floating_point():
                global_state_dict[key] += val * (n / total_samples)
            else:
                global_state_dict[key] += (val.float() * (n / total_samples)).to(val.dtype)
    return global_state_dict


def test_case_own_gm_model_summation(cfg):
    net = get_pytorch_model(cfg.model_name, cfg.model_cache_path,
                                deterministic=cfg.deterministic, channels=cfg.channels,  seed=cfg.seed)  
    temp_cache = Index(cfg.fw_cache_path)
    client_weights_nsamples = [
        temp_cache[f'cid_{i}'] for i in range(cfg.num_clients)]
    client_weights = [c[0] for c in client_weights_nsamples]
    client_nsamples = [c[1] for c in client_weights_nsamples]
    aggregated_state_dict = fedavg_aggregate(client_weights, client_nsamples)
    net.load_state_dict(aggregated_state_dict)
    total_model_weights = sum_model_weights_pytorch(net)
    return total_model_weights


def get_final_round_results(loss, accuracy, **args):
    test_case_pytorch_gm = args.get('pytorch_gm_sum', -1)
    sum_of_weights = args['framework_gm_sum']
    return {'Final Round Loss':loss, 'Final Round Accuracy':accuracy , 'PyTorch Local GM Sum':test_case_pytorch_gm, 'GM Framework Sum':sum_of_weights} 