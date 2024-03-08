# Package to preserve decoding test studies at Fermilab

## Dependencies

The needed packages are as follows:
- [Stim](https://github.com/quantumlib/Stim): You can get through pip by running `pip install stim` (with `~=1.12` version number if you need to specify it explicitly) and importing via `import stim`.
- [PyMatching](https://github.com/oscarhiggott/PyMatching) comes in a separate repository. You can install it through `pip install pymatching` and include it via `import pymatching`.
- Routines inside the notebooks use a few utilities from scikit-learn and scipy, and produce the plots using matplotlib. If you do not have any one of these, they can also be installed through pip: `pip install scikit-learn`, `pip install scipy`, and `pip install matplotlib`, respectively.
- You would also need Tensorflow (`pip install tensorflow`) and QKeras (`pip install qkeras`) for NNs.

## What is inside this repository?

- The [Stim playground notebook](https://github.com/usarica/FNAL-QCDecodingTests/blob/master/Stim_playground.ipynb) introduces a few different wrapper functions I wrote to make the collection of statistics and plotting easier. It also reproduces some of the exercises in the [Stim Getting Started notebook](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb).
  - A note on the Stim Getting Started notebook: Not everything is described there; some information may need to be searched within other documents in Stim the repository. For example, the meaning of different gates and other shorthands in the circuit printout are described [here](https://github.com/quantumlib/Stim/blob/main/doc/gates.md).

- The [notebook for distance-4 surface codes](https://github.com/usarica/FNAL-QCDecodingTests/blob/master/surface_code_d4.ipynb) outlines my progress during the week of Feb. 26 on initial studies for a simple surface code over which one could run different decoding algorithms.
  - In this notebook, I have been able to run PyMatching successfully.
  - It looks like running a [Stim-based BP+OSD implementation](https://github.com/oscarhiggott/stimbposd) takes a very long time for a very small number of shots on my personal computer, so I stopped that for now.
  - I was able to run an MLP with a few layers, testing NN decoder performance against PyMatching. When packed bit information about detection events is used, there does not appear to be a good edge over PyMatching.

- The [d=4, r=1 notebook](https://github.com/usarica/FNAL-QCDecodingTests/blob/master/surface_code_d4_r1.ipynb) includes studies during the week of March 4. Here, I followed the study by Google to include in NN training features both detection events and individual measurements that result in these detections. I use two dense layers + softmax output with labels being the d=4-bit observation information (in the end, I measure the fraction of mispredicted XORs of the unpacked predicted category bits, not the fraction of mispredicted category labels themselves). This seems to outperform PyMatching. There is a hint of overtraining at large number of nodes n for the same number of epochs, which is probably why n=120 in the middle performs better (if there is overtraining, I suppose it could be such that the performance over the XOR worsens).

In these studies, I assume the probabilities of all noise channels are the same, and their built-in implementation in Stim assumes they are constant over time.

## Relevant papers
- October '23 paper from Google: https://arxiv.org/abs/2310.05900
- Delft group's paper on NN decoders and hardware costs: https://arxiv.org/abs/2202.05741
