"""
GPU Batch Inference Server for MCTS training.

Architecture:
- A dedicated process runs the policy network on GPU
- Worker processes write leaf states to shared memory request slots
- The GPU process polls for requests, batches them, runs inference, writes results
- Workers poll for their result and continue tree traversal

Communication is entirely through shared memory (mp.Array) with atomic
slot flags, avoiding any IPC serialization overhead.

Slot lifecycle:
  0 = FREE (worker can claim)
  1 = WRITING (worker is filling request)
  2 = READY (request ready for GPU)
  3 = PROCESSING (GPU has picked it up)
  4 = DONE (result available for worker)
"""
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
import time
import ctypes


# Slot states
FREE = 0
WRITING = 1
READY = 2
PROCESSING = 3
DONE = 4

NUM_HEROES = 90


class GPUBatchServer:
    """Manages shared memory slots and the GPU inference process."""

    def __init__(self, network_state_dict, state_dim, num_slots=512,
                 max_batch=256, device='cuda:0'):
        self.state_dim = state_dim
        self.num_slots = num_slots
        self.max_batch = max_batch
        self.device_str = device

        # Shared memory layout per slot:
        # Request:  state (state_dim) + mask (NUM_HEROES) = state_dim + 90 floats
        # Response: priors (NUM_HEROES) + value (1) = 91 floats
        self.req_size = state_dim + NUM_HEROES
        self.resp_size = NUM_HEROES + 1

        # Allocate shared memory
        self.slot_flags = mp.Array(ctypes.c_int, num_slots, lock=False)
        self.req_buf = mp.Array(ctypes.c_float, num_slots * self.req_size, lock=False)
        self.resp_buf = mp.Array(ctypes.c_float, num_slots * self.resp_size, lock=False)

        # Network weights in shared memory (for updates)
        flat_sd = self._flatten_sd(network_state_dict)
        self.net_weights = mp.Array(ctypes.c_float, flat_sd, lock=False)
        self.net_shapes = [(k, list(v.shape)) for k, v in network_state_dict.items()]
        self.weights_version = mp.Value(ctypes.c_int, 0)

        # Control
        self.running = mp.Value(ctypes.c_bool, True)
        self.server_process = None

        # Stats
        self.total_batches = mp.Value(ctypes.c_long, 0)
        self.total_samples = mp.Value(ctypes.c_long, 0)

    def _flatten_sd(self, sd):
        return np.concatenate([v.cpu().numpy().flatten().astype(np.float32) for v in sd.values()])

    def _unflatten_sd(self, flat):
        sd = {}
        offset = 0
        for key, shape in self.net_shapes:
            numel = 1
            for s in shape:
                numel *= s
            sd[key] = torch.tensor(flat[offset:offset+numel]).reshape(shape)
            offset += numel
        return sd

    def start(self, network_class):
        """Start the GPU server process."""
        self.server_process = mp.Process(
            target=_gpu_server_loop,
            args=(network_class, self.state_dim, self.num_slots, self.max_batch,
                  self.device_str, self.slot_flags, self.req_buf, self.resp_buf,
                  self.req_size, self.resp_size, self.net_weights, self.net_shapes,
                  self.weights_version, self.running,
                  self.total_batches, self.total_samples),
            daemon=True,
        )
        self.server_process.start()

    def update_weights(self, state_dict):
        """Update network weights (called from main training process)."""
        flat = self._flatten_sd(state_dict)
        self.net_weights[:len(flat)] = flat
        self.weights_version.value += 1

    def shutdown(self):
        self.running.value = False
        if self.server_process:
            self.server_process.join(timeout=5)

    def stats(self):
        b = self.total_batches.value
        s = self.total_samples.value
        return {'batches': b, 'samples': s, 'avg_batch': s / b if b > 0 else 0}


def _gpu_server_loop(network_class, state_dim, num_slots, max_batch, device_str,
                     slot_flags, req_buf, resp_buf, req_size, resp_size,
                     net_weights, net_shapes, weights_version, running,
                     total_batches, total_samples):
    """GPU server process main loop."""
    device = torch.device(device_str)
    network = network_class().to(device)
    network.eval()

    # Load initial weights
    current_version = -1

    req_np = np.frombuffer(req_buf, dtype=np.float32).reshape(num_slots, req_size)
    resp_np = np.frombuffer(resp_buf, dtype=np.float32).reshape(num_slots, resp_size)

    while running.value:
        # Check for weight updates
        if weights_version.value != current_version:
            flat = np.array(net_weights[:], dtype=np.float32)
            sd = {}
            offset = 0
            for key, shape in net_shapes:
                numel = 1
                for s in shape:
                    numel *= s
                sd[key] = torch.tensor(flat[offset:offset+numel]).reshape(shape)
                offset += numel
            network.load_state_dict(sd)
            network.eval()
            current_version = weights_version.value

        # Collect ready requests
        batch_slots = []
        for i in range(num_slots):
            if slot_flags[i] == READY:
                slot_flags[i] = PROCESSING
                batch_slots.append(i)
                if len(batch_slots) >= max_batch:
                    break

        if not batch_slots:
            # Spin-wait with tiny sleep to avoid burning CPU
            time.sleep(0.00005)  # 50 microseconds
            continue

        # Extract batch from shared memory
        states = np.array([req_np[i, :state_dim] for i in batch_slots])
        masks = np.array([req_np[i, state_dim:] for i in batch_slots])

        # GPU forward pass
        s_t = torch.tensor(states, dtype=torch.float32).to(device)
        m_t = torch.tensor(masks, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, values = network(s_t, m_t)
            priors = F.softmax(logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()

        # Write results to shared memory
        for j, slot in enumerate(batch_slots):
            resp_np[slot, :NUM_HEROES] = priors[j]
            resp_np[slot, NUM_HEROES] = vals[j]
            slot_flags[slot] = DONE

        total_batches.value += 1
        total_samples.value += len(batch_slots)


class WorkerInferenceClient:
    """Used by worker processes to submit requests and read results."""

    def __init__(self, slot_flags, req_buf, resp_buf, state_dim, req_size, resp_size,
                 num_slots, worker_id):
        self.slot_flags = slot_flags
        self.req_np = np.frombuffer(req_buf, dtype=np.float32).reshape(num_slots, req_size)
        self.resp_np = np.frombuffer(resp_buf, dtype=np.float32).reshape(num_slots, resp_size)
        self.state_dim = state_dim
        self.num_slots = num_slots
        # Each worker gets a preferred slot range to reduce contention
        self.preferred_start = (worker_id * 4) % num_slots

    def predict(self, state_np, mask_np):
        """Submit inference request and block until result is ready.
        Returns (priors_np, value_float).
        """
        # Find a free slot
        slot = self._acquire_slot()

        # Write request
        self.req_np[slot, :self.state_dim] = state_np
        self.req_np[slot, self.state_dim:] = mask_np
        self.slot_flags[slot] = READY

        # Wait for result (spin-wait)
        while self.slot_flags[slot] != DONE:
            pass  # tight spin -- GPU latency is <1ms

        # Read result
        priors = self.resp_np[slot, :NUM_HEROES].copy()
        value = float(self.resp_np[slot, NUM_HEROES])
        self.slot_flags[slot] = FREE
        return priors, value

    def _acquire_slot(self):
        """Find and claim a free slot using compare-and-swap pattern."""
        start = self.preferred_start
        attempts = 0
        while True:
            for offset in range(self.num_slots):
                i = (start + offset) % self.num_slots
                if self.slot_flags[i] == FREE:
                    self.slot_flags[i] = WRITING
                    return i
            attempts += 1
            if attempts > 100:
                import time
                time.sleep(0.0001)  # back off if all slots busy
