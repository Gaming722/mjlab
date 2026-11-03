"""Passage scoring must reject falls and lateral escapes."""

import torch

from mjlab.scripts.evaluate_go1_stairs import passage_masks


def test_stair_passage_requires_height_and_no_previous_failure():
  position = torch.tensor([[3.5, 0.0, 0.0]]).repeat(4, 1)
  displacement = torch.tensor(
    [
      [3.5, 0.0, 1.0],
      [3.5, 0.0, 0.0],
      [3.5, 0.0, 1.0],
      [3.5, 0.0, 1.0],
    ]
  )
  position[3, 1] = 4.0
  passed, failed = passage_masks(
    position,
    displacement,
    torch.full((4,), -1.0),
    torch.full((4,), 0.9),
    torch.tensor([False, False, True, False]),
  )
  assert passed.tolist() == [True, False, False, False]
  assert failed.tolist() == [False, False, True, True]


def test_stair_passage_rejects_falling_and_supports_descent():
  position = torch.tensor([[3.5, 0.0, 0.0], [3.5, 0.0, 0.0]])
  displacement = torch.tensor([[3.5, 0.0, -1.0], [3.5, 0.0, -1.0]])
  passed, failed = passage_masks(
    position,
    displacement,
    torch.tensor([-1.0, 0.0]),
    torch.full((2,), -0.9),
    torch.zeros(2, dtype=torch.bool),
  )
  assert passed.tolist() == [True, False]
  assert failed.tolist() == [False, True]
