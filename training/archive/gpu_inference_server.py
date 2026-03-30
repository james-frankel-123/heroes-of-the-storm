"""
GPU Inference Server for batched MCTS leaf evaluation.

Receives batch requests from worker threads, concatenates into mega-batches,
runs single GPU forward pass, distributes results back.

Designed for virtual-loss batched MCTS where each worker submits K=32 states
at once, yielding GPU batches of 64-1024 samples.
"""
import threading
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


class BatchInferenceRequest:
    """A batch of states submitted by one worker."""
    __slots__ = ['states', 'masks', 'event', 'priors', 'values']

    def __init__(self, states: np.ndarray, masks: np.ndarray):
        self.states = states      # (K, state_dim)
        self.masks = masks        # (K, num_heroes)
        self.event = threading.Event()
        self.priors = None        # (K, num_heroes) — filled by server
        self.values = None        # (K,) — filled by server


class GPUInferenceServer:
    """Batches inference requests from multiple workers onto GPU."""

    def __init__(self, network, device='cuda:0', max_wait_ms=1.0):
        self.network = network.to(device)
        self.network.eval()
        self.device = torch.device(device)
        self.max_wait_ms = max_wait_ms

        self.queue = deque()
        self.lock = threading.Lock()
        self.has_work = threading.Event()
        self.running = True

        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

        self.total_batches = 0
        self.total_samples = 0

    def batch_predict(self, states: np.ndarray, masks: np.ndarray):
        """Blocking: submit batch of K states, wait for results.
        Returns (priors: (K, 90), values: (K,)).
        """
        req = BatchInferenceRequest(states, masks)
        with self.lock:
            self.queue.append(req)
        self.has_work.set()
        req.event.wait()
        return req.priors, req.values

    def _server_loop(self):
        while self.running:
            self.has_work.wait(timeout=0.005)
            self.has_work.clear()

            batch = []
            with self.lock:
                while self.queue:
                    batch.append(self.queue.popleft())

            if not batch:
                continue

            # Concatenate all requests into one mega-batch
            all_states = np.concatenate([r.states for r in batch])
            all_masks = np.concatenate([r.masks for r in batch])

            # Single GPU forward pass
            with torch.no_grad():
                s_t = torch.from_numpy(all_states).to(self.device)
                m_t = torch.from_numpy(all_masks).to(self.device)
                logits, values = self.network(s_t, m_t)
                all_priors = F.softmax(logits, dim=1).cpu().numpy()
                all_values = values.cpu().numpy().flatten()

            # Distribute results back to each request
            offset = 0
            for req in batch:
                k = len(req.states)
                req.priors = all_priors[offset:offset + k]
                req.values = all_values[offset:offset + k]
                offset += k
                req.event.set()

            self.total_batches += 1
            self.total_samples += len(all_states)

    def update_weights(self, state_dict):
        """Update network weights (called from main training loop)."""
        self.network.load_state_dict(state_dict)
        self.network.eval()

    def shutdown(self):
        self.running = False
        self.has_work.set()
        self.thread.join(timeout=5)

    def stats(self):
        avg = self.total_samples / self.total_batches if self.total_batches > 0 else 0
        return {
            'batches': self.total_batches,
            'samples': self.total_samples,
            'avg_batch_size': round(avg, 1),
        }
