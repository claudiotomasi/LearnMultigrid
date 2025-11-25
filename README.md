# LearnMultigrid

Python implementation accompanying the paper:

> **C. Tomasi, R. Krause**  
> *Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach*  
> arXiv:2109.05873  
> https://arxiv.org/abs/2109.05873

This repository provides the experimental code used to study how **deep neural networks can learn transfer operators** for multilevel / multigrid solvers.

The goal is to replace (or complement) expensive, problem-dependent constructions of restriction/prolongation operators with **data-driven operators** learned from examples of finite element mass matrices.

---

## Overview

Classical multigrid methods require carefully designed transfer operators between fine and coarse grids. These operators are usually derived analytically or geometrically and can be expensive or difficult to generalize.

In this work:

- We use **supervised deep learning** to learn the *coupling operator* $B_h$ that defines a pseudo-$L^2$ projection–based transfer operator $Q$.
- Once trained, the neural network predicts suitable operators directly from local information in the fine-grid **mass matrix** $M_h$.
- The learned transfer operators are inserted into a standard multigrid scheme and compared against a semi-geometric multigrid baseline.

The Python code implements:

- Data generation from 1D and 2D finite element meshes  
- Training of neural networks for 1D and 2D cases  
- Evaluation of the learned operators inside a multigrid solver  

For the full mathematical description and numerical results, please refer to the paper.

---

## Repository

In this repository you will find Python modules and scripts for:

- Generating training data from finite element mass matrices
- Defining and training neural networks (1D and 2D)
- Constructing transfer operators $Q$ from the predicted coupling operator $B_h$
- Using the learned operators inside a multigrid solver

---

## Requirements

- Python 3.7+  
- NumPy  
- SciPy  
- TensorFlow/Keras  
- Matplotlib  

If a `requirements.txt` file is present, install everything with:

```bash
pip install -r requirements.txt
```

---

## Getting Started

```bash
unzip LearnMultigrid.zip -d LearnMultigrid
cd LearnMultigrid
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python <script_name>.py
```

---

## Typical Workflow

### 1. **Generate training data**

Create examples $(x, y)$ where:

- $x$ = patches extracted from the fine-grid **mass matrix** $M_h$
- $y$ = corresponding rows/patches of the **coupling operator** $B_h$

### 2. **Train the neural network**

Train separate models for **1D** and **2D**:

- Loss combines:
  - Mean Squared Error (MSE)
  - Constraint-preserving regularization

### 3. **Build transfer operators**

From the predicted $B_h$, reconstruct:

- The transfer operator $Q$
- The coarse-grid operators (via Galerkin projection)

### 4. **Use in a multigrid solver**

Insert the learned $Q$ into a V-cycle and compare:

- Convergence rate  
- CPU time  

Compare against classical baselines.

---

## Reproducing Experiments

1. Generate datasets  
2. Train networks  
3. Run multigrid experiments  
4. Compare results  

---

## License

This code is available for **research and educational use**.

You may use, modify, and redistribute the code **provided that** you:

1. Acknowledge the original authors  
2. **Cite the associated paper** in any published work  

---

## How to Cite

```bibtex
@article{TomasiKrause2021NNMG,
  author  = {Tomasi, Claudio and Krause, Rolf},
  title   = {Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach},
  journal = {arXiv preprint arXiv:2109.05873},
  year    = {2021},
  url     = {https://arxiv.org/abs/2109.05873}
}
```

```bibtex
@incollection{TomasiKrause2021LNCS,
  author    = {Tomasi, Claudio and Krause, Rolf},
  title     = {Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach},
  booktitle = {Advances in High Performance Computing},
  editor    = {Fidanova, Stefka},
  series    = {Lecture Notes in Computer Science},
  volume    = {13209},
  publisher = {Springer},
  year      = {2021},
  doi       = {10.1007/978-3-030-95025-5_63}
}
```

---

## Author

**Claudio Tomasi**
