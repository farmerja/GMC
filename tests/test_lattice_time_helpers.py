import torch

from validation.lattice_time import build_structured_rect_problem, classify_face_and_snap


def test_build_structured_rect_problem_matches_refinement():
    problem = build_structured_rect_problem(refinement=4)
    assert problem.num_cells_x == 28
    assert problem.num_cells_y == 28
    assert problem.source_cells.shape[0] == 16


def test_classify_face_and_snap_without_corner_ties():
    x = torch.tensor([0.0, 1.0, 0.5, 0.5], dtype=torch.float32)
    y = torch.tensor([0.2, 0.8, 1.0, 0.0], dtype=torch.float32)
    W = torch.ones_like(x)
    H = torch.ones_like(y)
    face, x_snap, y_snap, pos = classify_face_and_snap(x, y, W, H)
    assert face.tolist() == [0, 1, 2, 3]
    assert torch.allclose(x_snap[:2], torch.tensor([0.0, 1.0]))
    assert torch.allclose(y_snap[2:], torch.tensor([1.0, 0.0]))
    assert torch.allclose(pos, torch.tensor([0.2, 0.8, 0.5, 0.5], dtype=torch.float32))
