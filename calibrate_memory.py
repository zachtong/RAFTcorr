import torch
import torch.nn.functional as F
import sys
import gc

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def simulate_raft_large(h, w):
    # Simulate RAFT-Large Correlation Volume
    # Input: H, W
    # Feature map: H/8, W/8
    # Channels: 256 (but correlation is dot product, so N*N)
    
    reset_memory()
    try:
        fmap_h, fmap_w = h // 8, w // 8
        N = fmap_h * fmap_w
        
        # Level 0: N x N (This is the killer)
        # In practice, RAFT computes: corr = torch.matmul(fmap1, fmap2)
        # fmap1: [1, 256, N]
        # Result: [1, N, N]
        
        # We simulate the allocation of the pyramid levels
        
        # Level 0
        sz0 = N * N * 4 # Float32 bytes
        t0 = torch.empty(N, N, dtype=torch.float32, device='cuda')
        
        # Level 1 (1/2 size in H, W -> 1/4 in area -> 1/16 in correlation size?)
        # Wait, Pooling is 2x2. 
        # Fmap size becomes (H/16, W/16) -> N/4 pixels.
        # Corr size becomes (N/4) * (N/4) = N^2 / 16.
        t1 = torch.empty(N//4, N//4, dtype=torch.float32, device='cuda')
        
        # Level 2
        t2 = torch.empty(N//16, N//16, dtype=torch.float32, device='cuda')
        
        # Level 3
        t3 = torch.empty(N//64, N//64, dtype=torch.float32, device='cuda')
        
        peak = get_memory_mb()
        del t0, t1, t2, t3
        return peak
    except torch.cuda.OutOfMemoryError:
        return -1
    except Exception as e:
        return -1

def simulate_raft_fine(h, w):
    # Simulate RAFT-Fine Correlation Volume
    # Input: H, W
    # Feature map: H, W (Full Res)
    
    reset_memory()
    try:
        N = h * w
        
        # Level 0: N x N
        t0 = torch.empty(N, N, dtype=torch.float32, device='cuda')
        
        # Level 1 (Pooling 2x2 -> N/4 pixels)
        # Corr: (N/4) * (N/4) = N^2 / 16
        t1 = torch.empty(N//4, N//4, dtype=torch.float32, device='cuda')
        
        peak = get_memory_mb()
        del t0, t1
        return peak
    except torch.cuda.OutOfMemoryError:
        return -1
    except Exception as e:
        return -1

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot calibrate.")
        sys.exit(0)
        
    print("Calibrating RAFT-Large (Standard)...")
    sizes_large = [256, 512, 1024]
    for sz in sizes_large:
        mem = simulate_raft_large(sz, sz)
        print(f"Size {sz}x{sz}: {mem:.2f} MB")
        
    print("\nCalibrating RAFT-Fine (Full Res)...")
    sizes_fine = [64, 128, 160] # Smaller sizes because it explodes fast
    for sz in sizes_fine:
        mem = simulate_raft_fine(sz, sz)
        print(f"Size {sz}x{sz}: {mem:.2f} MB")
