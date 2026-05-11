import jax

# Ensure this is True for 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import s2fft
import time
import logging

# 1a. (Optional) Silence those tracing warnings
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)

# 1b. Setup test parameters
L = 64
sampling = "gl" # <--- Changed from "mw" to "gl"
nrt = 1  # Number of round trips - REDUCED TO 1

# 1c. Force the precomputation onto the CPU
# We use jax.devices("cpu")[0] to ensure no TPU memory is touched yet
cpu_device = jax.devices("cpu")[0]
gpu_device = jax.devices("cuda")[0] # Changed from tpu_device

print("Precomputing weights on CPU (FP64)...")
with jax.default_device(cpu_device):
    # These are generated as high-precision JAX arrays on the host (CPU)
    pre_f_64 = s2fft.generate_precomputes_jax(L, forward=True, sampling=sampling)
    pre_i_64 = s2fft.generate_precomputes_jax(L, forward=False, sampling=sampling)

# 2. Cast to FP32 and move to GPU
# This step "cleans" the data for the GPU hardware (note s2fft returns a list,
# so we use tree_map to apply the cast and movement to every array in the list)

print("Casting and uploading to GPU...")
pre_f = jax.tree_util.tree_map(lambda x: jax.device_put(x.astype(jnp.float64), gpu_device), pre_f_64)
pre_i = jax.tree_util.tree_map(lambda x: jax.device_put(x.astype(jnp.float64), gpu_device), pre_i_64)

# 3. Define the Round Trip function
@jax.jit
def round_trip(signal, weights_f, weights_i):
    # Use the high-level wrapper with method="jax"
    # This ensures the logic for reality and sampling is handled correctly
    flm = s2fft.forward(signal, L, sampling=sampling,
                        method="jax", precomps=weights_f, reality=True)

    f_rec = s2fft.inverse(flm, L, sampling=sampling,
                          method="jax", precomps=weights_i, reality=True)
    return flm, f_rec # Return flm as well


# 4.  Initialize a simple signal for Gauss-Lobatto (constant function)
# For 'gl' sampling with reality=True, the grid size for the output is (L, 2*L-1)
N_PIXELS = L * (2 * L - 1) # Updated for Gauss-Lobatto sampling grid (L, 2L-1)
f_reference = jnp.ones((L, 2 * L - 1), dtype=jnp.float64) # Changed to a constant function and correct shape
# Ensure the reference signal is on the GPU
f_reference = jax.device_put(f_reference, gpu_device)
f_comp_init = f_reference # Initialize the f_comp (we will compare to reference at the end)

# 5. Benchmark section
print(f"Benchmarking...")

# 5a. Measure Compilation (First Run)
print("Compiling (Tracing + XLA)...")
t0 = time.perf_counter()
flm_warmup, f_rec_warmup = round_trip(f_comp_init, pre_f, pre_i)
flm_warmup.block_until_ready()
f_rec_warmup.block_until_ready()
compile_time = time.perf_counter() - t0
print(f"Compilation finished in: {compile_time:.4f}s")

# 5b. Timing Loop
print(f"Starting {nrt} round trips...")

start_time = time.time()
current_f = f_comp_init
for i in range(nrt):
    flm_out, current_f = round_trip(current_f, pre_f, pre_i)

# Wait for GPU to finish the last calculation
current_f = current_f.block_until_ready()
flm_out = flm_out.block_until_ready()

total_time = time.time() - start_time
avg_trip = total_time / nrt

# Calculate Max reconstruction error
max_err = jnp.abs(f_reference - current_f).max()

print("\n" + "="*35)
print(f"GPU BENCHMARK RESULTS") # Updated from TPUv5
print(f"Sampling: {sampling}") # <--- Added sampling info
print(f"L: {L}, N_PIXELS: {N_PIXELS}") # <--- Updated grid info
print(f"Avg Trip Time: {avg_trip:.6e}s")
print(f"Max Recon Error: {max_err:.6e}") # Changed to Max error
print(f"Min f_comp: {jnp.min(current_f):.6e}") # Added min f_comp
print(f"Max f_comp: {jnp.max(current_f):.6e}") # Added max f_comp

# Inspect flm coefficients
# For s2fft with reality=True, L=0, M=0 is at flm[0, L-1]
theoretical_f00 = jnp.sqrt(4 * jnp.pi)

print(f"flm (0,0) coefficient: {flm_out[0, L-1]:.6e}")
print(f"Theoretical flm (0,0): {theoretical_f00:.6e}")
print(f"Abs diff flm (0,0): {jnp.abs(flm_out[0, L-1] - theoretical_f00):.6e}")

# Check other l=0, m!=0 coefficients
l0_non_m0_mask = jnp.arange(flm_out.shape[1]) != (L-1)
max_abs_l0_non_m0 = jnp.max(jnp.abs(flm_out[0, l0_non_m0_mask]))
print(f"Max absolute value of l=0, m!=0 coefficients: {max_abs_l0_non_m0:.6e}")

# Check l>0 coefficients
max_abs_l_gt_0 = jnp.max(jnp.abs(flm_out[1:, :]))
print(f"Max absolute value of l>0 coefficients: {max_abs_l_gt_0:.6e}")

print("="*35)
