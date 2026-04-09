#!/usr/bin/env python3
"""
Validation script for 1D time-dependent pulse transport problem.

Implements Fully Asynchronous Time Evaluation:
- Particles possess individual clocks and advance across time exactly like standard MC.
- Includes a Hybrid MC+CFM Solver: MC natively handles all free streaming, and queues
  particles that scatter into batched Deep CFM Neural evaluations.
- Includes a Pure CFM Solver: ML handles all internal and boundary transport.
- Steps are mathematically bounded by exact time bin boundaries to ensure 
  accurate, non-smeared track-length flux tallying.

Creates animations showing the time evolution of the scalar flux profile.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.mc1dtime import simulate_1d_slab_time_dependent_full
from ml.transform import (
    normalize_thickness, normalize_position_relative, normalize_uv, normalize_u_in,
    normalize_log_scale_logit, denorm_linear_scale_logit, denorm_uv,
    denorm_log_scale_logit
)
from ml.model import WrappedModel, get_model
from flow_matching.solver import ODESolver
from validation.utils import find_model_file, load_norm_params_file, decode_budgeted_length


# =============================================================================
# Configuration
# =============================================================================

EPSILON = 1e-9
HIDDEN_DIM, DEPTH = 160, 4

INTERNAL_MODEL_DIR = "ml/models/1d_timedep_internal"
INTERNAL_NORM_FILE = "ml/data/norm_params_1d_timedep_internal.npy"
INTERNAL_INPUT_DIM, INTERNAL_COND_DIM = 7, 4  # [thick, budget, x_in, mu_in] -> [x_out, mu_out, L]

BOUNDARY_MODEL_DIR = "ml/models/1d_timedep_boundary"
BOUNDARY_NORM_FILE = "ml/data/norm_params_1d_timedep_boundary.npy"
BOUNDARY_INPUT_DIM, BOUNDARY_COND_DIM = 6, 3  # [thick, budget, mu_in] -> [x_out, mu_out, L]


# =============================================================================
# CFM Model Loading
# =============================================================================

def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_grad_enabled(False)
    return device


def get_compute_dtype(device):
    return torch.float32 if device == "mps" else torch.float64


def load_models(device, load_boundary=False):
    models = {}
    norm_params = {}

    if Path(INTERNAL_MODEL_DIR).exists():
        try:
            internal_model = get_model(INTERNAL_INPUT_DIM, INTERNAL_COND_DIM, HIDDEN_DIM, DEPTH).to(device)
            internal_model_path = find_model_file(INTERNAL_MODEL_DIR)
            internal_model.load_state_dict(torch.load(internal_model_path, map_location=device))
            internal_model.eval()
            internal_wrapper = WrappedModel(internal_model).to(device)
            internal_solver = ODESolver(velocity_model=internal_wrapper)
            models['internal'] = {'model': internal_model, 'solver': internal_solver}
        except Exception as exc:
            print(f"WARNING: Failed to load internal model: {exc}")

        try:
            in_path, loaded = load_norm_params_file(Path(INTERNAL_NORM_FILE))
            norm_params['internal'] = {
                'thick_lower': loaded.get('thick_lower', loaded.get('thickness_lower')),
                'thick_upper': loaded.get('thick_upper', loaded.get('thickness_upper')),
                'budget_lower': loaded['budget_lower'], 'budget_upper': loaded['budget_upper'],
                'L_lower': loaded['L_lower'], 'L_upper': loaded['L_upper'],
                'use_L_over_budget': bool(loaded.get('use_L_over_budget', False)),
            }
        except Exception as exc:
            print(f"WARNING: Failed to load internal norm params: {exc}")

    if load_boundary and Path(BOUNDARY_MODEL_DIR).exists():
        try:
            boundary_model = get_model(BOUNDARY_INPUT_DIM, BOUNDARY_COND_DIM, HIDDEN_DIM, DEPTH).to(device)
            boundary_model_path = find_model_file(BOUNDARY_MODEL_DIR)
            boundary_model.load_state_dict(torch.load(boundary_model_path, map_location=device))
            boundary_model.eval()
            boundary_wrapper = WrappedModel(boundary_model).to(device)
            boundary_solver = ODESolver(velocity_model=boundary_wrapper)
            models['boundary'] = {'model': boundary_model, 'solver': boundary_solver}
        except Exception as exc:
            print(f"WARNING: Failed to load boundary model: {exc}")

        try:
            bnd_path, loaded = load_norm_params_file(Path(BOUNDARY_NORM_FILE))
            norm_params['boundary'] = {
                'thick_lower': loaded.get('thick_lower', loaded.get('thickness_lower')),
                'thick_upper': loaded.get('thick_upper', loaded.get('thickness_upper')),
                'budget_lower': loaded['budget_lower'], 'budget_upper': loaded['budget_upper'],
                'L_lower': loaded['L_lower'], 'L_upper': loaded['L_upper'],
                'use_L_over_budget': bool(loaded.get('use_L_over_budget', False)),
            }
        except Exception as exc:
            print(f"WARNING: Failed to load boundary norm params: {exc}")

    return models, norm_params


# =============================================================================
# Asynchronous Helper & Tallying Methods
# =============================================================================

def init_source(num_particles, source_type, domain_length, source_half_width, source_sigma, device):
    center_x = 0.5 * domain_length

    def _sample_mu_isotropic(n: int) -> torch.Tensor:
        mu_l = 2.0 * torch.rand(n, device=device) - 1.0
        return torch.where(torch.abs(mu_l) < EPSILON, torch.where(torch.rand_like(mu_l) > 0.5, EPSILON, -EPSILON), mu_l)

    if source_type == 0:
        x = torch.zeros(num_particles, device=device, dtype=torch.float32)
        mu = torch.sqrt(torch.rand(num_particles, device=device)).clamp(min=EPSILON)
    elif source_type == 1:
        x = torch.rand(num_particles, device=device) * domain_length
        mu = _sample_mu_isotropic(num_particles)
    elif source_type == 2:
        x = torch.full((num_particles,), center_x, device=device, dtype=torch.float32)
        mu = _sample_mu_isotropic(num_particles)
    elif source_type == 3:
        half_w = max(float(source_half_width), 0.0)
        left = max(0.0, center_x - half_w)
        right = min(domain_length, center_x + half_w)
        x = torch.full((num_particles,), center_x, device=device, dtype=torch.float32) if right <= left + EPSILON else left + torch.rand(num_particles, device=device, dtype=torch.float32) * (right - left)
        mu = _sample_mu_isotropic(num_particles)
    elif source_type == 4:
        sig = max(float(source_sigma), 0.0)
        x = torch.clamp(center_x + sig * torch.randn(num_particles, device=device, dtype=torch.float32), min=0.0, max=domain_length)
        mu = _sample_mu_isotropic(num_particles)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
        
    return x, mu


def _scatter_add_2d(cell_idx, time_bin, values, target):
    num_cells, num_bins = target.shape
    valid = (
        (cell_idx >= 0)
        & (cell_idx < num_cells)
        & (time_bin >= 0)
        & (time_bin < num_bins)
    )
    if valid.any():
        flat_idx = cell_idx[valid].to(torch.long) * num_bins + time_bin[valid].to(torch.long)
        target.view(-1).index_add_(0, flat_idx, values[valid].to(target.dtype))


def tally_flux_to_cell_async(cell_idx, time_bin, weight_start, path_length_cm, sigma_a, flux_tally, dx):
    optical_depth = sigma_a * path_length_cm
    frac = -torch.expm1(-optical_depth)
    track_add = weight_start * frac / sigma_a if sigma_a > 1e-12 else weight_start * path_length_cm
    _scatter_add_2d(cell_idx, time_bin, track_add / dx, flux_tally)


def tally_energy_deposition_async(cell_idx, time_bin, energy_deposited, energy_tally, dx):
    _scatter_add_2d(cell_idx, time_bin, energy_deposited / dx, energy_tally)


def tally_leakage_async(time_bin, weight, leakage_tally):
    valid = (time_bin >= 0) & (time_bin < leakage_tally.numel())
    if valid.any():
        leakage_tally.index_add_(0, time_bin[valid].to(torch.long), weight[valid].to(leakage_tally.dtype))


# =============================================================================
# Hybrid Vectorized Transport: Asynchronous MC + CFM
# =============================================================================

def run_hybrid_pulse_simulation_async(models, norm_params, device, num_particles, num_cells,
                             domain_length, sigma_t, sigma_s, total_time, num_time_bins, 
                             source_type=2, source_half_width=0.5, source_sigma=0.5,
                             num_ode_steps=12, max_iterations=5000, weight_cutoff=1e-6, max_scatters=1):
    
    if 'internal' not in models: return None, None, None, None, None
    internal_solver, int_norm = models['internal']['solver'], norm_params['internal']
    compute_dtype = get_compute_dtype(device)

    sigma_a = sigma_t - sigma_s
    dt = total_time / num_time_bins
    cell_width = domain_length / num_cells
    particle_velocity = 1.0
    cell_thickness = cell_width * sigma_s
    path_budget_phys_max = dt * particle_velocity

    tally_dtype = torch.float32 if device == "mps" else torch.float64
    flux_t = torch.zeros((num_cells, num_time_bins), device=device, dtype=tally_dtype)
    energy_deposition_t = torch.zeros((num_cells, num_time_bins), device=device, dtype=tally_dtype)
    leakage_left_t = torch.zeros(num_time_bins, device=device, dtype=tally_dtype)
    leakage_right_t = torch.zeros(num_time_bins, device=device, dtype=tally_dtype)

    T_grid = torch.linspace(0, 1, num_ode_steps + 1, device=device)
    step_size = T_grid[1] - T_grid[0]
    total_cfm_calls, total_mc_steps, total_scatters = 0, 0, 0
    budget_history_chunks = {'internal': []}

    print(f"Running Asynchronous Hybrid MC + CFM (Queue Triggers after {max_scatters} Scatter(s))...")

    x, mu = init_source(num_particles, source_type, domain_length, source_half_width, source_sigma, device)
    weight = torch.ones(num_particles, device=device, dtype=torch.float32)
    cell_idx = (x / cell_width).long().clamp(0, num_cells - 1)
    
    current_time = torch.zeros(num_particles, device=device, dtype=compute_dtype)
    current_tbin = torch.zeros(num_particles, device=device, dtype=torch.int32)
    status = torch.zeros(num_particles, device=device, dtype=torch.int8)  
    
    scatter_count = torch.zeros(num_particles, device=device, dtype=torch.int32)
    queued_for_cfm = torch.zeros(num_particles, device=device, dtype=torch.bool)

    pbar = tqdm(total=num_particles, desc="Simulating Particles")
    particles_completed = 0

    for iteration in range(max_iterations):
        active_mask = (status == 0)
        if not active_mask.any(): break

        # --- PHASE 1: Asynchronous Priority MC Loop ---
        while True:
            mc_mask = (status == 0) & (~queued_for_cfm)
            if not mc_mask.any(): break
            
            idx = mc_mask.nonzero(as_tuple=True)[0]

            # Enforce hard steps natively tracking to the bin boundary to remove tally smearing 
            tbin = current_tbin[idx]
            next_t = ((tbin + 1).to(compute_dtype) * dt)
            d_time = (next_t - current_time[idx]).clamp_min(0.0) * particle_velocity
            d_max = (total_time - current_time[idx]).clamp_min(0.0) * particle_velocity
            d_time = torch.minimum(d_time, d_max)

            # Prevent float stalling
            tiny = d_time < 1e-7
            if tiny.any():
                current_time[idx[tiny]] = next_t[tiny]
                current_tbin[idx[tiny]] += 1
                expired = current_tbin[idx[tiny]] >= num_time_bins
                if expired.any(): status[idx[tiny][expired]] = 2
                
                idx = idx[~tiny]
                if idx.numel() == 0: continue
                tbin, next_t, d_time = tbin[~tiny], next_t[~tiny], d_time[~tiny]

            n_mc = idx.numel()
            total_mc_steps += n_mc

            d_scat = (-torch.log(torch.rand(n_mc, device=device).clamp_min(1e-12)) / sigma_s).to(compute_dtype) if sigma_s > EPSILON else torch.full((n_mc,), float('inf'), device=device, dtype=compute_dtype)

            c_idx, c_left = cell_idx[idx], (cell_idx[idx].to(compute_dtype) * cell_width)
            c_right = c_left + cell_width
            mu_mc, x_mc = mu[idx].to(compute_dtype), x[idx].to(compute_dtype)

            d_bnd = torch.where(mu_mc > 0, (c_right - x_mc) / mu_mc.clamp_min(1e-12),
                    torch.where(mu_mc < 0, (c_left - x_mc) / mu_mc.clamp_max(-1e-12), torch.full((n_mc,), float('inf'), device=device, dtype=compute_dtype))).clamp_min(0.0)

            d_step = torch.minimum(torch.minimum(d_scat, d_bnd), d_time)

            hit_time = (d_step == d_time)
            hit_bnd = (d_step == d_bnd) & (~hit_time)
            hit_scat = (d_step == d_scat) & (~hit_time) & (~hit_bnd)

            x[idx] = torch.clamp(x_mc + mu_mc * d_step, c_left, c_right).to(torch.float32)
            
            d_step_f32 = d_step.to(torch.float32)
            weight_start = weight[idx].clone()
            weight[idx] *= torch.exp(-sigma_a * d_step_f32)

            tally_flux_to_cell_async(c_idx, tbin, weight_start, d_step_f32, sigma_a, flux_t, cell_width)
            tally_energy_deposition_async(c_idx, tbin, weight_start - weight[idx], energy_deposition_t, cell_width)

            current_time[idx] = torch.where(hit_time, next_t, current_time[idx] + d_step / particle_velocity)
            current_tbin[idx] = torch.where(hit_time, tbin + 1, tbin)

            if hit_bnd.any():
                bnd_idx, mu_bnd = idx[hit_bnd], mu_mc[hit_bnd]
                x[bnd_idx] = torch.where(mu_bnd > 0, c_right[hit_bnd].float(), c_left[hit_bnd].float())
                
                new_cell = c_idx[hit_bnd] + torch.where(mu_bnd > 0, 1, -1)
                cell_idx[bnd_idx] = new_cell
                scatter_count[bnd_idx] = 0
                
                exited = (new_cell < 0) | (new_cell >= num_cells)
                if exited.any():
                    e_idx, new_c_e = bnd_idx[exited], new_cell[exited]
                    
                    leak_left = new_c_e < 0
                    if leak_left.any(): tally_leakage_async(tbin[hit_bnd][exited][leak_left], weight[e_idx[leak_left]], leakage_left_t)
                    leak_right = new_c_e >= num_cells
                    if leak_right.any(): tally_leakage_async(tbin[hit_bnd][exited][leak_right], weight[e_idx[leak_right]], leakage_right_t)
                    status[e_idx] = 1

            if hit_scat.any():
                scat_idx = idx[hit_scat]
                scatter_count[scat_idx] += 1
                total_scatters += scat_idx.numel()
                
                mu_new = 2.0 * torch.rand(scat_idx.numel(), device=device) - 1.0
                mu[scat_idx] = torch.where(torch.abs(mu_new) < EPSILON, torch.where(torch.rand_like(mu_new) > 0.5, EPSILON, -EPSILON), mu_new)
                queued_for_cfm[scat_idx[scatter_count[scat_idx] >= max_scatters]] = True

            low_weight = weight[idx] < weight_cutoff
            if low_weight.any(): 
                status[idx[low_weight]] = 2
                queued_for_cfm[idx[low_weight]] = False
            
            done = current_time[idx] >= (total_time - 1e-12)
            if done.any():
                status[idx[done]] = 2
                queued_for_cfm[idx[done]] = False

        # --- PHASE 2: Asynchronous Deep Cell Batched CFM ---
        cfm_mask = (status == 0) & queued_for_cfm
        if cfm_mask.any():
            idx = cfm_mask.nonzero(as_tuple=True)[0]
            
            tbin = current_tbin[idx]
            next_t = ((tbin + 1).to(compute_dtype) * dt)
            d_time = (next_t - current_time[idx]).clamp_min(0.0) * particle_velocity
            d_max = (total_time - current_time[idx]).clamp_min(0.0) * particle_velocity
            d_time = torch.minimum(d_time, d_max)

            budget_phys = torch.clamp(d_time, max=path_budget_phys_max).to(torch.float32)

            tiny = budget_phys < 1e-7
            if tiny.any():
                current_time[idx[tiny]] = next_t[tiny]
                current_tbin[idx[tiny]] += 1
                expired = current_tbin[idx[tiny]] >= num_time_bins
                if expired.any(): status[idx[tiny][expired]] = 2
                queued_for_cfm[idx[tiny]] = False
                
                idx = idx[~tiny]
                if idx.numel() == 0: continue
                tbin, next_t, d_time, budget_phys = tbin[~tiny], next_t[~tiny], d_time[~tiny], budget_phys[~tiny]

            n_cfm = idx.numel()
            total_cfm_calls += n_cfm

            c_idx = cell_idx[idx]
            c_left = c_idx.float() * cell_width
            x_local = ((x[idx] - c_left) / cell_width).clamp(0.001, 0.999)
            
            rem_bud_opt = budget_phys * sigma_s if sigma_s > EPSILON else budget_phys
            budget_history_chunks['internal'].append(rem_bud_opt.detach())

            cond = torch.stack([
                normalize_thickness(torch.full((n_cfm,), cell_thickness, device=device), int_norm['thick_lower'], int_norm['thick_upper']),
                normalize_log_scale_logit(rem_bud_opt.clamp(min=1e-6), int_norm['budget_lower'], int_norm['budget_upper']),
                normalize_position_relative(x_local), normalize_uv(mu[idx])
            ], dim=1)

            with torch.no_grad():
                out = internal_solver.sample(time_grid=T_grid, x_init=torch.cat([cond, torch.randn(n_cfm, 3, device=device)], dim=1), method='rk4', step_size=step_size, return_intermediates=False)

            pred = out[:, INTERNAL_COND_DIM:]
            x_out_local = denorm_linear_scale_logit(pred[:, 0]).clamp(0, 1)
            mu_out = denorm_uv(pred[:, 1]).clamp(-1, 1)
            L_out_opt = decode_budgeted_length(pred[:, 2], rem_bud_opt, int_norm).clamp_min(0)
            L_out_opt = torch.minimum(L_out_opt, rem_bud_opt)

            L_out_phys = L_out_opt / sigma_s if sigma_s > EPSILON else L_out_opt
            x_pred_global = c_left + x_out_local * cell_width
            dist_geom_phys = torch.abs(x_pred_global - x[idx])
            
            scale = torch.where(dist_geom_phys > budget_phys, budget_phys / dist_geom_phys.clamp_min(1e-12), torch.ones_like(dist_geom_phys))
            x_out_local = (((x[idx] + (x_pred_global - x[idx]) * scale) - c_left) / cell_width).clamp(0.0, 1.0)

            space_exit = (x_out_local <= 1e-4) | (x_out_local >= 1.0 - 1e-4) & (~(dist_geom_phys > budget_phys))
            L_out_phys = torch.where(~space_exit, budget_phys, L_out_phys)
            L_out_phys = torch.maximum(torch.minimum(L_out_phys, budget_phys), dist_geom_phys * scale)
            mu_out = torch.where(space_exit & (x_out_local < 0.5), -mu_out.abs(), torch.where(space_exit & (x_out_local >= 0.5), mu_out.abs(), mu_out))

            weight_start = weight[idx].clone()
            weight[idx] *= torch.exp(-sigma_a * L_out_phys)

            tally_flux_to_cell_async(c_idx, tbin, weight_start, L_out_phys, sigma_a, flux_t, cell_width)
            tally_energy_deposition_async(c_idx, tbin, weight_start - weight[idx], energy_deposition_t, cell_width)

            hit_full_bin = (~space_exit) & (torch.abs(budget_phys.to(compute_dtype) - d_time) < 1e-9)
            current_time[idx] = torch.where(hit_full_bin, next_t, current_time[idx] + L_out_phys.to(compute_dtype) / particle_velocity)
            current_tbin[idx] = torch.where(hit_full_bin, tbin + 1, tbin)

            expired = current_tbin[idx] >= num_time_bins
            if expired.any(): status[idx[expired]] = 2

            new_x = c_left + x_out_local * cell_width
            x[idx] = torch.where(space_exit & (x_out_local >= 0.5), c_left + cell_width, torch.where(space_exit & (x_out_local < 0.5), c_left, new_x))
            mu[idx] = mu_out

            new_cell = c_idx + torch.where(space_exit & (x_out_local >= 0.5), 1, torch.where(space_exit & (x_out_local < 0.5), -1, 0))
            cell_idx[idx] = new_cell

            exited = (new_cell < 0) | (new_cell >= num_cells)
            if exited.any():
                e_idx, new_c_e = idx[exited], new_cell[exited]
                
                leak_left = new_c_e < 0
                if leak_left.any(): tally_leakage_async(tbin[exited][leak_left], weight[e_idx[leak_left]], leakage_left_t)
                leak_right = new_c_e >= num_cells
                if leak_right.any(): tally_leakage_async(tbin[exited][leak_right], weight[e_idx[leak_right]], leakage_right_t)
                status[e_idx] = 1

            queued_for_cfm[idx] = False
            scatter_count[idx] = 0
            
            low_weight = weight[idx] < weight_cutoff
            if low_weight.any(): status[idx[low_weight]] = 2
            done = current_time[idx] >= (total_time - 1e-12)
            if done.any(): status[idx[done]] = 2

        newly_completed = (~(status == 0)).sum().item() - particles_completed
        if newly_completed > 0:
            pbar.update(newly_completed)
            particles_completed += newly_completed

    pbar.close()
    flux = flux_t.detach().cpu().numpy()
    energy_deposition = energy_deposition_t.detach().cpu().numpy()
    leakage_left = leakage_left_t.detach().cpu().numpy()
    leakage_right = leakage_right_t.detach().cpu().numpy()
    budget_history = {}
    for key, chunks in budget_history_chunks.items():
        if chunks:
            budget_history[key] = torch.cat(chunks, dim=0).detach().cpu().numpy().tolist()
        else:
            budget_history[key] = []
    print(f"\n[Hybrid Debug Statistics]\n  MC stream loops evaluated: {total_mc_steps} | Scatters triggered: {total_scatters} | Batched CFM calls: {total_cfm_calls}")

    return flux, leakage_left, leakage_right, energy_deposition, budget_history


# =============================================================================
# Pure CFM Vectorized Transport: Asynchronous Evaluation
# =============================================================================

def run_pure_cfm_pulse_simulation_async(models, norm_params, device, num_particles, num_cells,
                             domain_length, sigma_t, sigma_s, total_time, num_time_bins, 
                             source_type=2, source_half_width=0.5, source_sigma=0.5,
                             num_ode_steps=12, max_iterations=2000, weight_cutoff=1e-6):

    int_solver, bnd_solver = models['internal']['solver'], models['boundary']['solver']
    int_norm, bnd_norm = norm_params['internal'], norm_params['boundary']
    compute_dtype = get_compute_dtype(device)

    sigma_a, dt, cell_width = sigma_t - sigma_s, total_time / num_time_bins, domain_length / num_cells
    cell_thickness = cell_width * sigma_s
    path_budget_phys_max = dt * 1.0

    tally_dtype = torch.float32 if device == "mps" else torch.float64
    flux_t = torch.zeros((num_cells, num_time_bins), device=device, dtype=tally_dtype)
    energy_deposition_t = torch.zeros((num_cells, num_time_bins), device=device, dtype=tally_dtype)
    leakage_left_t = torch.zeros(num_time_bins, device=device, dtype=tally_dtype)
    leakage_right_t = torch.zeros(num_time_bins, device=device, dtype=tally_dtype)

    T_grid = torch.linspace(0, 1, num_ode_steps + 1, device=device)
    step_size = T_grid[1] - T_grid[0]
    total_int_calls, total_bnd_calls = 0, 0
    budget_history_chunks = {'internal': [], 'boundary': []}

    print("Running Asynchronous Pure CFM Simulation...")

    x, mu = init_source(num_particles, source_type, domain_length, source_half_width, source_sigma, device)
    weight = torch.ones(num_particles, device=device, dtype=torch.float32)
    cell_idx = (x / cell_width).long().clamp(0, num_cells - 1)
    
    current_time = torch.zeros(num_particles, device=device, dtype=compute_dtype)
    current_tbin = torch.zeros(num_particles, device=device, dtype=torch.int32)
    status = torch.zeros(num_particles, device=device, dtype=torch.int8)
    
    x_local_init = ((x - cell_idx.float() * cell_width) / cell_width)
    is_on_boundary = (x_local_init <= 1e-4) | (x_local_init >= 1.0 - 1e-4)

    pbar = tqdm(total=num_particles, desc="Simulating Particles")
    particles_completed = 0

    for iteration in range(max_iterations):
        active_mask = (status == 0)
        if not active_mask.any(): break

        for is_bnd, solver, norm, name, dim in [(True, bnd_solver, bnd_norm, 'boundary', BOUNDARY_COND_DIM), (False, int_solver, int_norm, 'internal', INTERNAL_COND_DIM)]:
            mask = active_mask & is_on_boundary if is_bnd else active_mask & ~is_on_boundary
            if not mask.any(): continue
            
            idx = mask.nonzero(as_tuple=True)[0]

            tbin = current_tbin[idx]
            next_t = ((tbin + 1).to(compute_dtype) * dt)
            d_time = (next_t - current_time[idx]).clamp_min(0.0) * 1.0
            d_max = (total_time - current_time[idx]).clamp_min(0.0) * 1.0
            d_time = torch.minimum(d_time, d_max)

            budget_phys = torch.clamp(d_time, max=path_budget_phys_max).to(torch.float32)

            tiny = budget_phys < 1e-7
            if tiny.any():
                current_time[idx[tiny]] = next_t[tiny]
                current_tbin[idx[tiny]] += 1
                expired = current_tbin[idx[tiny]] >= num_time_bins
                if expired.any(): status[idx[tiny][expired]] = 2
                
                idx = idx[~tiny]
                if idx.numel() == 0: continue
                tbin, next_t, d_time, budget_phys = tbin[~tiny], next_t[~tiny], d_time[~tiny], budget_phys[~tiny]
                
            n_eval = idx.numel()
            if is_bnd: total_bnd_calls += n_eval
            else: total_int_calls += n_eval

            c_idx, c_left = cell_idx[idx], cell_idx[idx].float() * cell_width
            x_local = ((x[idx] - c_left) / cell_width).clamp(0.001, 0.999)

            rem_bud_opt = budget_phys * sigma_s if sigma_s > EPSILON else budget_phys
            budget_history_chunks[name].append(rem_bud_opt.detach())

            if is_bnd:
                mu_abs = mu[idx].abs().clamp(min=EPSILON)
                input_neg = mu[idx] < 0
                cond = torch.stack([
                    normalize_thickness(torch.full((n_eval,), cell_thickness, device=device), norm['thick_lower'], norm['thick_upper']),
                    normalize_log_scale_logit(rem_bud_opt.clamp(min=1e-6), norm['budget_lower'], norm['budget_upper']),
                    normalize_u_in(mu_abs)
                ], dim=1)
            else:
                cond = torch.stack([
                    normalize_thickness(torch.full((n_eval,), cell_thickness, device=device), norm['thick_lower'], norm['thick_upper']),
                    normalize_log_scale_logit(rem_bud_opt.clamp(min=1e-6), norm['budget_lower'], norm['budget_upper']),
                    normalize_position_relative(x_local), normalize_uv(mu[idx])
                ], dim=1)

            with torch.no_grad():
                out = solver.sample(time_grid=T_grid, x_init=torch.cat([cond, torch.randn(n_eval, 3, device=device)], dim=1), method='rk4', step_size=step_size, return_intermediates=False)
            
            pred = out[:, dim:]
            x_out_local = denorm_linear_scale_logit(pred[:, 0]).clamp(0, 1)
            mu_out = denorm_uv(pred[:, 1]).clamp(-1, 1)
            L_out_opt = decode_budgeted_length(pred[:, 2], rem_bud_opt, norm).clamp_min(0)
            L_out_opt = torch.minimum(L_out_opt, rem_bud_opt)

            L_out_phys = L_out_opt / sigma_s if sigma_s > EPSILON else L_out_opt

            if is_bnd:
                x_out_local = torch.where(input_neg, 1.0 - x_out_local, x_out_local)
                mu_out = torch.where(input_neg, -mu_out, mu_out)

            dist_geom_phys = torch.abs(c_left + x_out_local * cell_width - x[idx])
            scale = torch.where(dist_geom_phys > budget_phys, budget_phys / dist_geom_phys.clamp_min(1e-12), torch.ones_like(dist_geom_phys))
            x_out_local = (((x[idx] + (c_left + x_out_local * cell_width - x[idx]) * scale) - c_left) / cell_width).clamp(0.0, 1.0)
            
            space_exit = (x_out_local <= 1e-4) | (x_out_local >= 1.0 - 1e-4) & (~(dist_geom_phys > budget_phys))
            L_out_phys = torch.where(~space_exit, budget_phys, L_out_phys)
            L_out_phys = torch.maximum(torch.minimum(L_out_phys, budget_phys), dist_geom_phys * scale)

            weight_start = weight[idx].clone()
            weight[idx] *= torch.exp(-sigma_a * L_out_phys)
            tally_flux_to_cell_async(c_idx, tbin, weight_start, L_out_phys, sigma_a, flux_t, cell_width)
            tally_energy_deposition_async(c_idx, tbin, weight_start - weight[idx], energy_deposition_t, cell_width)

            hit_full_bin = (~space_exit) & (torch.abs(budget_phys.to(compute_dtype) - d_time) < 1e-9)
            current_time[idx] = torch.where(hit_full_bin, next_t, current_time[idx] + L_out_phys.to(compute_dtype))
            current_tbin[idx] = torch.where(hit_full_bin, tbin + 1, tbin)
            
            expired = current_tbin[idx] >= num_time_bins
            if expired.any(): status[idx[expired]] = 2

            new_x = c_left + x_out_local * cell_width
            x[idx] = torch.where(space_exit & (x_out_local >= 0.5), c_left + cell_width, torch.where(space_exit & (x_out_local < 0.5), c_left, new_x))
            mu[idx] = torch.where(space_exit & (x_out_local < 0.5), -mu_out.abs(), torch.where(space_exit & (x_out_local >= 0.5), mu_out.abs(), mu_out))
            is_on_boundary[idx] = space_exit

            new_cell = c_idx + torch.where(space_exit & (x_out_local >= 0.5), 1, torch.where(space_exit & (x_out_local < 0.5), -1, 0))
            cell_idx[idx] = new_cell

            exited = (new_cell < 0) | (new_cell >= num_cells)
            if exited.any():
                e_idx, new_c_e = idx[exited], new_cell[exited]
                leak_left = new_c_e < 0
                if leak_left.any(): tally_leakage_async(tbin[exited][leak_left], weight[e_idx[leak_left]], leakage_left_t)
                leak_right = new_c_e >= num_cells
                if leak_right.any(): tally_leakage_async(tbin[exited][leak_right], weight[e_idx[leak_right]], leakage_right_t)
                status[e_idx] = 1
                
            low_weight = weight[idx] < weight_cutoff
            if low_weight.any(): status[idx[low_weight]] = 2
            done = current_time[idx] >= (total_time - 1e-12)
            if done.any(): status[idx[done]] = 2

        newly_completed = (~(status == 0)).sum().item() - particles_completed
        if newly_completed > 0:
            pbar.update(newly_completed)
            particles_completed += newly_completed

    pbar.close()
    flux = flux_t.detach().cpu().numpy()
    energy_deposition = energy_deposition_t.detach().cpu().numpy()
    leakage_left = leakage_left_t.detach().cpu().numpy()
    leakage_right = leakage_right_t.detach().cpu().numpy()
    budget_history = {}
    for key, chunks in budget_history_chunks.items():
        if chunks:
            budget_history[key] = torch.cat(chunks, dim=0).detach().cpu().numpy().tolist()
        else:
            budget_history[key] = []
    print(f"\n[Pure CFM Debug Statistics]\n  Total internal CFM calls: {total_int_calls} | Total boundary CFM calls: {total_bnd_calls}")
    return flux, leakage_left, leakage_right, energy_deposition, budget_history


# =============================================================================
# Visualization Diagnostics
# =============================================================================

def create_flux_animation(flux_dict, x_centers, times, output_path, fps=10, log_scale=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    start_col = 3 if len(times) > 3 else 0
    flux_max = max(f[:, start_col:].max() for f in flux_dict.values()) * 1.1
    L = x_centers[-1] + (x_centers[1] - x_centers[0]) / 2

    lines = {}
    colors = {'Ref MC': ('g-', 3, 0.7), 'MC': ('b-', 2, 1.0), 'Pure CFM': ('m--', 2, 1.0), 'Hybrid CFM': ('r--', 2, 1.0)}
    for name, data in flux_dict.items():
        style, lw, alpha = colors.get(name, ('k-', 1, 1.0))
        lines[name] = (ax.plot([], [], style, linewidth=lw, alpha=alpha, label=name)[0], data)

    ax.set_xlim(0, L)
    if log_scale:
        ax.set_yscale('log')
        flux_min = min(f[f > 0].min() for f in flux_dict.values() if (f > 0).any()) * 0.5
        ax.set_ylim(flux_min, flux_max * 2)
        ax.set_ylabel('Flux (log scale)', fontsize=12)
    else:
        ax.set_ylim(0, flux_max)
        ax.set_ylabel('Flux', fontsize=12)
    ax.set_xlabel('Position x', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    def init():
        for line, _ in lines.values(): line.set_data([], [])
        time_text.set_text('')
        return [line for line, _ in lines.values()] + [time_text]

    def animate(frame):
        for line, data in lines.values():
            y_data = data[:, frame].copy()
            if log_scale: y_data = np.maximum(y_data, 1e-10)
            line.set_data(x_centers, y_data)
        time_text.set_text(f't = {times[frame]:.2f}')
        return [line for line, _ in lines.values()] + [time_text]

    animation.FuncAnimation(fig, animate, init_func=init, frames=len(times), interval=1000/fps, blit=True).save(output_path, writer='pillow', fps=fps)
    plt.close()


def plot_flux_snapshots(flux_dict, x_centers, times, output_path, snapshot_indices=None, log_scale=False):
    if snapshot_indices is None:
        n_snaps = min(6, len(times))
        snapshot_indices = np.linspace(0, len(times) - 1, n_snaps, dtype=int)

    fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(4*len(snapshot_indices), 4))
    if len(snapshot_indices) == 1: axes = [axes]

    L = x_centers[-1] + (x_centers[1] - x_centers[0]) / 2
    flux_max = max(f[:, snapshot_indices].max() for f in flux_dict.values()) * 1.1
    colors = {'Ref MC': ('g-', 3, 0.7), 'MC': ('b-', 2, 1.0), 'Pure CFM': ('m--', 2, 1.0), 'Hybrid CFM': ('r--', 2, 1.0)}

    for i, snap_idx in enumerate(snapshot_indices):
        ax = axes[i]
        for name, data in flux_dict.items():
            y_data = np.maximum(data[:, snap_idx], 1e-10) if log_scale else data[:, snap_idx]
            style, lw, alpha = colors.get(name, ('k-', 1, 1.0))
            ax.plot(x_centers, y_data, style, linewidth=lw, alpha=alpha, label=name)
            
        ax.set_title(f't = {times[snap_idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_xlim(0, L)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(min(f[f > 0].min() for f in flux_dict.values() if (f > 0).any()) * 0.5, flux_max * 2)
        else:
            ax.set_ylim(0, flux_max)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_budget_distribution(budget_history, norm_params, output_path):
    keys = [k for k in ['internal', 'boundary'] if k in budget_history and len(budget_history[k]) > 0]
    if not keys: return
        
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 5))
    if len(keys) == 1: axes = [axes]

    for idx, name in enumerate(keys):
        ax = axes[idx]
        budgets = np.array(budget_history[name])
        budget_lower = 10 ** norm_params[name]['budget_lower']
        budget_upper = 10 ** norm_params[name]['budget_upper']

        log_budgets = np.log10(np.clip(budgets, 1e-10, None))
        ax.hist(log_budgets, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(np.log10(budget_lower), color='red', linestyle='--', linewidth=2)
        ax.axvline(np.log10(budget_upper), color='red', linestyle='--', linewidth=2)

        ax.set_title(f'{name.capitalize()} Model Budgets')
        ax.set_xlabel('log₁₀(Budget)', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_flux_centroid_stats(label, flux_norm, x_centers):
    xbar = (x_centers[:, None] * flux_norm).sum(axis=0) / (flux_norm.sum(axis=0) + 1e-30)
    print(f"{label} xbar min/max: {xbar.min():.6f}, {xbar.max():.6f}")


# =============================================================================
# Main Sequence
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='1D Pulse Transport Validation (Asynchronous Time Evaluation)')
    parser.add_argument('--num_particles', type=int, default=50000)
    parser.add_argument('--num_cells', type=int, default=27)
    parser.add_argument('--domain_length', type=float, default=10.0)
    parser.add_argument('--sigma_t', type=float, default=1.0)
    parser.add_argument('--sigma_s', type=float, default=0.5)
    parser.add_argument('--total_time', type=float, default=4.0)
    parser.add_argument('--num_time_bins', type=int, default=50)
    parser.add_argument('--source_type', type=int, default=2)
    parser.add_argument('--source_half_width', type=float, default=0.5)
    parser.add_argument('--source_sigma', type=float, default=0.5)
    parser.add_argument('--max_scatters', type=int, default=1, help='Hybrid Only: Scatters to trigger CFM queue')
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'pure_cfm', 'mc'], help='Solver mode')
    parser.add_argument('--output_dir', type=str, default='validation/results/pulse_1d_async')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--mc_only', action='store_true')
    parser.add_argument('--run_pure_cfm', action='store_true', help="Run the asynchronous Pure CFM model alongside the Hybrid")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_device()

    print("\n=== 1D Pulse Transport Validation (Asynchronous Execution) ===")
    dx = args.domain_length / args.num_cells
    dt = args.total_time / args.num_time_bins
    x_centers = np.linspace(dx/2, args.domain_length - dx/2, args.num_cells)
    times = np.linspace(dt/2, args.total_time - dt/2, args.num_time_bins)
    
    models, norm_params = {}, {}
    if not args.mc_only and args.mode != 'mc':
        print("\n--- Loading CFM Models ---")
        models, norm_params = load_models(device, load_boundary=(args.run_pure_cfm or args.mode == 'pure_cfm'))

    ref_np = args.num_particles * 10
    print(f"\n--- Running Reference MC Simulation ({ref_np:,} particles) ---")
    np.random.seed(42)  
    ref_flux, *_ = simulate_1d_slab_time_dependent_full(
        num_particles=ref_np, slab_thickness=args.domain_length, sigma_t=args.sigma_t, sigma_s=args.sigma_s,
        particle_velocity=1.0, max_time=args.total_time, num_time_bins=args.num_time_bins, num_space_bins=args.num_cells, 
        source_type=args.source_type, source_half_width=args.source_half_width, source_sigma=args.source_sigma
    )
    ref_flux_norm = ref_flux / ref_np

    print(f"\n--- Running Target Fast MC Simulation ({args.num_particles:,} particles) ---")
    np.random.seed(123)
    mc_flux, *_ = simulate_1d_slab_time_dependent_full(
        num_particles=args.num_particles, slab_thickness=args.domain_length, sigma_t=args.sigma_t, sigma_s=args.sigma_s,
        particle_velocity=1.0, max_time=args.total_time, num_time_bins=args.num_time_bins, num_space_bins=args.num_cells, 
        source_type=args.source_type, source_half_width=args.source_half_width, source_sigma=args.source_sigma
    )
    mc_flux_norm = mc_flux / args.num_particles
    print_flux_centroid_stats("MC", mc_flux_norm, x_centers)

    flux_dict = {'Ref MC': ref_flux_norm, 'MC': mc_flux_norm}

    if not args.mc_only and args.mode != 'mc':
        if (args.mode == 'pure_cfm' or args.run_pure_cfm) and 'internal' in models and 'boundary' in models:
            cfm_flux, _, _, _, _ = run_pure_cfm_pulse_simulation_async(
                models, norm_params, device, args.num_particles, args.num_cells, args.domain_length, 
                args.sigma_t, args.sigma_s, args.total_time, args.num_time_bins, args.source_type, 
                args.source_half_width, args.source_sigma
            )
            if cfm_flux is not None:
                flux_dict['Pure CFM'] = cfm_flux / args.num_particles
                print_flux_centroid_stats("Pure CFM", flux_dict['Pure CFM'], x_centers)

        if args.mode == 'hybrid' and 'internal' in models:
            hybrid_flux, _, _, _, b_hist = run_hybrid_pulse_simulation_async(
                models, norm_params, device, args.num_particles, args.num_cells, args.domain_length, 
                args.sigma_t, args.sigma_s, args.total_time, args.num_time_bins, args.source_type, 
                args.source_half_width, args.source_sigma, max_scatters=args.max_scatters
            )
            if hybrid_flux is not None:
                flux_dict['Hybrid CFM'] = hybrid_flux / args.num_particles
                print_flux_centroid_stats("Hybrid CFM", flux_dict['Hybrid CFM'], x_centers)
                plot_budget_distribution(b_hist, norm_params, os.path.join(args.output_dir, 'budget_distribution.png'))

    # Plot Outputs
    create_flux_animation(flux_dict, x_centers, times, os.path.join(args.output_dir, 'comparison_animation.gif'), fps=args.fps, log_scale=False)
    create_flux_animation(flux_dict, x_centers, times, os.path.join(args.output_dir, 'comparison_animation_log.gif'), fps=args.fps, log_scale=True)
    plot_flux_snapshots(flux_dict, x_centers, times, os.path.join(args.output_dir, 'snapshots.png'), log_scale=False)
    plot_flux_snapshots(flux_dict, x_centers, times, os.path.join(args.output_dir, 'snapshots_log.png'), log_scale=True)
    print(f"\nAll validations successfully saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
