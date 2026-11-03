"""
GMC cell kernels for validation/chain_core.chain (2D electrons, square cells).

    BaselineFlowKernel   the 2d_electron flow (exit position predicted as a free 2D point); optionally an exit
                         classifier picks the class and the flow is redrawn until it agrees (rejection_match)
    FaceFlowKernel       exit classifier picks the class (face / absorbed); the 2d_electron_face flow predicts the
                         state on that face (position along it, outgoing angle, energy, path)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.cell_topology import (SQUARE, UNIT_SQUARE_VERTS, polygon_face_point, direction_from_face_frame_2d,
                              polygon_nearest_face)
from ml.exit_classifier import load_exit_classifier, check_norm_compatibility, rejection_match
from ml.model import get_model
from ml.transform import (load_normalization_params, normalize_position_relative, normalize_angle_phi,
                          normalize_radius_r, normalize_log_scale_probit, uv_to_polar, polar_to_uv,
                          denorm_position_relative, denorm_angle_phi, denorm_radius_r, denorm_linear_scale_probit,
                          denorm_exit_angle, decode_energy_out)
from ml.utils import run_uniform_ode_inference

NF = SQUARE.num_faces
ABSORBED = SQUARE.interior_class("absorbed")


def load_flow(ckpt_path, device):
    """Flow checkpoint -> (model, cond_dim); architecture read from the weights."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    hidden, input_dim = sd["input_proj.weight"].shape
    target_dim = sd["output.weight"].shape[0]
    depth = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    expansion = sd["blocks.0.ff.0.weight"].shape[0] // hidden
    model = get_model(input_dim, input_dim - target_dim, hidden_dim=hidden, depth=depth, expansion_factor=expansion)
    model.load_state_dict(sd)
    return model.to(device).eval(), input_dim - target_dim


def electron_cond(y_c, u_c, v_c, E, norm_params):
    """The 5 conditioning features shared by every 2D electron model (canonical frame, x_in = 0)."""
    phi, r = uv_to_polar(u_c, v_c)
    return np.stack([
        normalize_position_relative(y_c),
        normalize_angle_phi(phi),
        normalize_radius_r(r),
        normalize_log_scale_probit(E, float(norm_params["E_in_lower"]), float(norm_params["E_in_upper"])),
        normalize_position_relative(np.zeros_like(y_c)),
    ], axis=1)


class _FlowKernelBase:
    def __init__(self, device, ode_steps=100, ode_method="euler", seed=0):
        self.device, self.ode_steps, self.ode_method = device, ode_steps, ode_method
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def _ode(self, model, cond, target_dim):
        noise = torch.randn(cond.shape[0], target_dim, generator=self.gen)
        x0 = torch.cat([torch.as_tensor(cond, dtype=torch.float32), noise], 1).to(self.device)
        z = run_uniform_ode_inference(model, x0, self.device, self.ode_method, self.ode_steps, show_timing=False)
        return z[:, cond.shape[1]:].double().numpy()

    def _w_from_r(self, r):
        # Models carry only |w| (through r = sqrt(1 - w^2)); the sign is irrelevant by z-symmetry, so draw it.
        return self.rng.choice([-1.0, 1.0], r.shape[0]) * np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))


class BaselineFlowKernel(_FlowKernelBase):
    """2d_electron flow. Absorbed if its E_out is at the cutoff: f_out <= stop_f_out when given, else
    E_out <= E_cut + stop_tol (the original Macro_field rule, in MeV)."""

    def __init__(self, ckpt, norm_path, device, classifier=None, stop_f_out=None, stop_tol=1e-4, max_tries=16,
                 **kw):
        super().__init__(device, **kw)
        self.norm, _ = load_normalization_params(norm_path)
        self.model, self.cond_dim = load_flow(ckpt, device)
        self.E_cut = float(self.norm["E_cutoff"])
        self.stop_f_out, self.stop_tol, self.max_tries = stop_f_out, stop_tol, max_tries
        self.classifier = None
        if classifier:
            clf, topo, ck = load_exit_classifier(classifier, device)
            check_norm_compatibility(ck, self.norm, ["E_in_lower", "E_in_upper"])
            self.classifier = clf
        self.unmatched, self.sampled = 0, 0

    def _sample(self, cond, E, rows):
        z = self._ode(self.model, cond[rows], 7)
        u, v = polar_to_uv(denorm_angle_phi(z[:, 2]), denorm_radius_r(z[:, 3]))
        f_out = np.clip(denorm_linear_scale_probit(z[:, 4]), 0.0, 1.0)
        return {"x": denorm_position_relative(z[:, 0]), "y": denorm_position_relative(z[:, 1]), "u": u, "v": v,
                "r": denorm_radius_r(z[:, 3]), "f_out": f_out,
                "E_out": np.maximum(f_out * (E[rows] - self.E_cut) + self.E_cut, self.E_cut)}

    def _cls(self, s):
        stopped = (s["f_out"] <= self.stop_f_out) if self.stop_f_out is not None else \
            (s["E_out"] <= self.E_cut + self.stop_tol)
        face = polygon_nearest_face(UNIT_SQUARE_VERTS, np.stack([s["x"], s["y"]], 1))
        return np.where(stopped, ABSORBED, face)

    def __call__(self, y_c, u_c, v_c, w, E, step):
        cond = electron_cond(y_c, u_c, v_c, E, self.norm)
        rows = np.arange(len(E))
        if self.classifier is None:
            s = self._sample(cond, E, rows)
        else:
            target = self.classifier.sample(torch.as_tensor(cond, dtype=torch.float32, device=self.device),
                                            generator=None).cpu().numpy()
            s, unmatched = rejection_match(target, lambda r: self._sample(cond, E, r), self._cls, self.max_tries)
            self.unmatched += int(unmatched.sum())
            self.sampled += len(E)
        return {"cls": self._cls(s), "x": s["x"], "y": s["y"], "u": s["u"], "v": s["v"],
                "w": self._w_from_r(s["r"]), "E_out": s["E_out"]}


class FaceFlowKernel(_FlowKernelBase):
    """Exit classifier + face-conditioned flow (2d_electron_face)."""

    def __init__(self, ckpt, norm_path, classifier, device, **kw):
        super().__init__(device, **kw)
        self.norm, _ = load_normalization_params(norm_path)
        self.model, self.cond_dim = load_flow(ckpt, device)
        self.clf, topo, ck = load_exit_classifier(classifier, device)
        check_norm_compatibility(ck, self.norm, ["E_in_lower", "E_in_upper"])
        if topo.class_names != SQUARE.class_names:
            raise ValueError(f"classifier classes {topo.class_names} != {SQUARE.class_names}")
        if self.cond_dim != 5 + NF:
            raise ValueError(f"face flow cond_dim {self.cond_dim} != {5 + NF}")
        self.E_cut = float(self.norm["E_cutoff"])

    def __call__(self, y_c, u_c, v_c, w, E, step):
        n = len(E)
        cond = electron_cond(y_c, u_c, v_c, E, self.norm)
        cls = self.clf.sample(torch.as_tensor(cond, dtype=torch.float32, device=self.device)).cpu().numpy()
        out = {"cls": cls, "x": np.zeros(n), "y": np.zeros(n), "u": np.zeros(n), "v": np.zeros(n), "w": np.zeros(n),
               "E_out": np.zeros(n)}
        ex = np.flatnonzero(cls < NF)
        if ex.size:
            onehot = np.eye(NF)[cls[ex]]
            z = self._ode(self.model, np.concatenate([cond[ex], onehot], 1), 5)
            face = cls[ex]
            s = np.clip(denorm_position_relative(z[:, 0]), 0.0, 1.0)
            psi = denorm_exit_angle(z[:, 1])
            r = denorm_radius_r(z[:, 2])
            p = polygon_face_point(UNIT_SQUARE_VERTS, face, s)
            u, v = direction_from_face_frame_2d(UNIT_SQUARE_VERTS, face, r * np.cos(psi), r * np.sin(psi))
            out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex] = p[:, 0], p[:, 1], u, v
            out["w"][ex] = self._w_from_r(r)
            out["E_out"][ex] = decode_energy_out(z[:, 3], E[ex], self.E_cut, self.norm)
        return out
