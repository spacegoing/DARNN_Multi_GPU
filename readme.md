## Acknowledgment ##

This repo is based on https://github.com/Zhenye-Na/DA-RNN.

## Improvements ##

- `DataSet` `DataLoader` implementation; easier training loop
- Batch training
- GPU support; Multi-GPUs support currently not working due to
  pytorch 0.4's bug
- Corrections of differences between original code and paper

## Multi-GPUs BUG [Solved] ##
Solved in pytorch 1.0. Hooray for users!

~Pytorch 0.4 Bug relates to this issue: https://github.com/pytorch/pytorch/issues/7092~

~`RuntimeError: torch/csrc/autograd/variable.cpp:115: get_grad_fn: Assertion `output_nr == 0` failed.`~

