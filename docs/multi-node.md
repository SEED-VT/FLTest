# Running on a multi-node cluster

By default every backend simulates all federated clients on the machine that runs `fltest`.
The Flower backend can instead spread its clients over a multi-node
[Ray](https://docs.ray.io) cluster, so a run with hundreds of clients uses every node's CPUs.
The reference and NVFlare backends always run on one machine.

## 1. Prepare identical nodes

Install FLTest on every node **at the same path, with the same Python environment**. Ray
workers import `fltest` and the client hooks from the driver's `sys.path`, so the virtual
environment and the repository must sit at the same location everywhere. For example,
`~/FLTest/.venv` on each node.

## 2. Start a Ray cluster

Use the `ray` binary from that environment. On the head node:

```bash
ray start --head --node-ip-address=<head-ip> --port=6379 --num-cpus=<cores> --disable-usage-stats
```

Then on every other node:

```bash
ray start --address=<head-ip>:6379 --node-ip-address=<this-node-ip> --num-cpus=<cores> --disable-usage-stats
```

`ray status` on the head should list every node and the total CPUs.

## 3. Run FLTest against the cluster

On the head node, point FLTest at the cluster and run any config whose runs use
`framework: flwr`:

```bash
export FLTEST_RAY_ADDRESS=<head-ip>:6379
fltest run my_config.yaml
```

With `FLTEST_RAY_ADDRESS` set, three things change:

- **Flower joins the cluster.** It connects to it instead of starting a private Ray sized by
  `total_cpus`/`total_gpus`, so those two settings are ignored and the cluster's resources
  apply. Each client still requests one CPU.
- **Data shards are kept in memory.** Prepared shards are held in memory rather than
  memory-mapped from the local Hugging Face cache, so they reach clients on other nodes. A
  memory-mapped table pickles as a file path that exists only on the driver node. These
  in-memory shards are cached under their own key, separate from single-machine runs.
- **Each shard is stored in the cluster once.** Before the run starts, each client's shard is
  put into Ray's object store, and a client fetches only its own. Flower sends the client app
  with every message to a client, so an app that held the shards would carry every client's
  data each time. With 120 CIFAR-10 clients that is about 800 MB per message, which fills the
  head node's disk.

Unset the variable to go back to single-machine runs.

## When it pays off

Flower starts one client worker per cluster CPU and ships each client's shard to its node.
For a handful of clients that overhead dominates, and a single machine is faster. The cluster
helps when there are many clients, or each client trains for many steps.
