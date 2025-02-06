# CONCORD: Stable and fault-tolerant decentralized WAN traffic engineering

**Dependencies:** `gurobipy`, `networkx`, `numpy`, `pickle`

## File summary:
- `solver.py` is a library file containing all TE optimization solvers, for all classical baselines and Concord-TE for all objectives
- `large-scale-experiments.py` is the script used to run TE experiments for each topology
- `permutation_experiments.py` is the script used to run the permutation slicing experiments from S8
- `data` folder contains the topology data for open-source topologies. (Data for KDL to be uploaded soon, due to the larger file sizes)
- `partition.py` contains the Concord-Slicer network partitioning algorithm
- `util.py` contains useful helpfer functions that are access across all scripts
- `NetworkParser.py` contains setup and all topology/demand parsing functions
- `graph_partitioning_solver.py` contains the ILP graph partitioning solver referenced in Appendix F

The code to implement DOTE can be found [here](https://github.com/PredWanTE/DOTE).

We cannot yet release PWAN topology/demand data, or the empirical demand deviations distribution for confidentiality reasons. We will release more code, including the exact data used to generate figures in the paper, for the artifact evaluation.
