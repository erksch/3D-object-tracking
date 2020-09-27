import torch 
from evaluation import mapping_from_overlap

def test_mapping_from_overlap():
    print("Testing mapping_from_overlap...")

    o_indices = [11, 12, 13, 14]
    h_indices = [1, 2, 3, 4, 5]
    overlap = torch.Tensor([
      [0.4, 0.2, 0.0, 0.0, 0.0], 
      [0.5, 0.4, 0.0, 0.0, 0.0],
      [0.0, 0.1, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    mapping = mapping_from_overlap(overlap, o_indices, h_indices)

    assert mapping == {
      11: 1,
      12: 2,
      14: 5,
    }

    print("Passed.")

if __name__ == '__main__':
  test_mapping_from_overlap()
