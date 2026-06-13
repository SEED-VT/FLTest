"""NVFlare client training script, launched per site by the simulator's ScriptRunner.

Runs in its own process, so FLTest client-side hooks do not apply here (documented
limitation of the NVFlare backend). It reconstructs the model/data from the run cache,
trains locally each round on the received global model, and returns the update.

Invoked as:  python client_script.py --client_id <id> --cache_path <dir>
"""

import argparse
import os

import torch
from diskcache import Index

import nvflare.client as flare

from fltest.data.models import get_model, test, train
from fltest.data.utils import seed_everything

os.environ.setdefault("PYTHONHASHSEED", "786")


def main(client_id: int, cache_path: str) -> None:
    cache = Index(cache_path)
    spec = cache["nvflare_spec"]
    dataset_dict = cache["nvflare_dataset"]
    seed_everything(spec.seed)

    shard = dataset_dict["c2data"][client_id]
    trainloader = torch.utils.data.DataLoader(
        shard, batch_size=spec.client_batch_size, shuffle=True, num_workers=0
    )

    net = get_model(spec.model_name, spec.model_cache_path, channels=spec.channels,
                    num_classes=spec.num_classes, deterministic=spec.deterministic).to(spec.device)

    flare.init()
    while flare.is_running():
        seed_everything(spec.seed)
        input_model = flare.receive()
        net.load_state_dict(input_model.params)
        net.to(spec.device)
        train(net, trainloader, epochs=spec.client_epochs, device=spec.device,
              loss_fn=spec.loss_fn, optimizer_name=spec.optimizer, lr=spec.client_lr)
        n_steps = spec.client_epochs * len(trainloader)
        output = flare.FLModel(
            params=net.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": n_steps},
        )
        flare.send(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    args = parser.parse_args()
    main(args.client_id, args.cache_path)
