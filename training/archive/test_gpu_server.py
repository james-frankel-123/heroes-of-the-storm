"""Benchmark GPU batch inference server vs CPU inference."""
import sys
import os
import time
import multiprocessing as mp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_draft_policy import AlphaZeroDraftNet, DraftState, STATE_DIM, NUM_HEROES
from gpu_batch_server import GPUBatchServer, WorkerInferenceClient

_worker_client = {}


def _bench_init(slot_flags, req_buf, resp_buf, state_dim):
    _worker_client['client'] = WorkerInferenceClient(
        slot_flags, req_buf, resp_buf,
        state_dim, state_dim + 90, 90 + 1, 512,
        worker_id=os.getpid() % 512,
    )


def _bench_work(n_iters):
    s = DraftState('Cursed Hollow', 'mid', our_team=0)
    state_np = s.to_numpy()
    mask_np = s.valid_mask_np()
    for _ in range(n_iters):
        _worker_client['client'].predict(state_np, mask_np)


def main():
    mp.set_start_method('spawn', force=True)

    network = AlphaZeroDraftNet()
    network.load_state_dict(torch.load('training/draft_policy.pt', weights_only=True, map_location='cpu'))
    network.eval()

    # CPU baseline
    state = DraftState('Cursed Hollow', 'mid', our_team=0)
    x = torch.tensor(state.to_numpy(), dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(state.valid_mask_np(), dtype=torch.float32).unsqueeze(0)
    t0 = time.time()
    for _ in range(500):
        with torch.no_grad():
            network(x, m)
    cpu_rate = 500 / (time.time() - t0)
    print(f"CPU single: {cpu_rate:.0f} samples/s")

    # Start GPU server
    server = GPUBatchServer(network.state_dict(), STATE_DIM, num_slots=512, max_batch=256)
    server.start(AlphaZeroDraftNet)
    time.sleep(3)

    # Single-process validation
    client = WorkerInferenceClient(
        server.slot_flags, server.req_buf, server.resp_buf,
        STATE_DIM, STATE_DIM + 90, 90 + 1, 512, worker_id=0,
    )
    priors, value = client.predict(state.to_numpy(), state.valid_mask_np())
    print(f"GPU single: value={value:.3f}, priors sum={priors.sum():.3f}")

    # Multi-worker benchmark
    for n_workers in [8, 16, 32, 64]:
        pool = mp.Pool(n_workers, initializer=_bench_init,
                       initargs=(server.slot_flags, server.req_buf, server.resp_buf, STATE_DIM))
        n_iters = 500
        t0 = time.time()
        pool.map(_bench_work, [n_iters] * n_workers)
        wall = time.time() - t0
        total = n_workers * n_iters
        rate = total / wall
        print(f"GPU {n_workers} workers: {rate:.0f} samples/s ({rate/cpu_rate:.1f}x vs CPU)")
        pool.close()
        pool.join()

    print(f"\nServer stats: {server.stats()}")
    server.shutdown()


if __name__ == '__main__':
    main()
