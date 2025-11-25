# LearnMultigrid

Python implementation accompanying the paper:

> **C. Tomasi, R. Krause**  

> *Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach*  

> arXiv:2109.05873  

> https://arxiv.org/abs/2109.05873

This repository provides the experimental code used to study how **deep neural networks can learn transfer operators** for multilevel / multigrid solvers.

The goal is to replace (or complement) expensive, problem–dependent constructions of restriction / prolongation operators with **data‑driven operators** learned from examples of finite element mass matrices.

---

## Overview

Classical multigrid methods require carefully designed transfer operators between fine and coarse grids. These operators are typically derived analytically or geometrically and can be expensive or difficult to generalize.

In this work:

- We use **supervised deep learning** to learn the *coupling operator* \(B_h\) that defines a pseudo–\(L^2\) projection–based transfer operator \(Q\).
- Once trained, the neural network predicts suitable operators directly from local information in the **fine‑grid mass matrix**.
- The learned transfer operators are plugged into a standard multigrid scheme and compared against a semi‑geometric multigrid baseline.

The Python code implements:

- **Data generation** from 1D and 2D finite element meshes  
- **Training** of neural networks for 1D and 2D cases  
- **Evaluation** of the learned operators inside a multigrid solver  

For the full mathematical description and numerical results, please refer to the paper.

---

## Repository

In this repository you will find Python modules and scripts for:

- Generating training records from finite element mass matrices
- Defining and training neural network models (1D and 2D)
- Constructing transfer operators \(Q\) from the predicted coupling operator \(B_h\)
- Using the learned operators inside a multigrid (MG) solver and comparing against a reference method


---

## Requirements

The implementation is written in **Python** and relies on the standard scientific stack plus a deep learning framework.

Typical requirements are:

- Python 3.7+  
- NumPy  
- SciPy  
- TensorFlow / Keras (for the neural networks)  
- Matplotlib (for plots of loss, error, and convergence)

If a `requirements.txt` file is present in the archive, you can install everything with:

```bash
pip install -r requirements.txt
```

Otherwise, install the above packages manually.

---

## Getting Started

1. **Clone or download** this repository and locate the ZIP archive.
2. **Extract** the ZIP archive, e.g.:

   ```bash
   unzip LearnMultigrid.zip -d LearnMultigrid
   cd LearnMultigrid
   ```

3. (Optional but recommended) **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies** (if `requirements.txt` is available):

   ```bash
   pip install -r requirements.txt
   ```

5. Inspect the available Python scripts and notebooks (for example, training scripts for 1D and 2D models, data generation utilities, and MG test routines), and run them with:

   ```bash
   python <script_name>.py
   ```

---

## Typical Workflow

While details depend on the provided scripts, the general workflow is:

1. **Generate training data**

   Create examples \((x, y)\) where:
   - Features \(x\) are patches extracted from the fine‑grid **mass matrix** \(M_h\).
   - Targets \(y\) are corresponding rows/patches of the **coupling operator** \(B_h\).

2. **Train the neural network**

   - Use regression with a loss function that combines:
     - Mean Squared Error (MSE)
     - Regularization terms enforcing problem‑specific constraints (e.g., preservation of constants, closeness to the true operator as described in the paper).
   - Train separate models for **1D** and **2D** meshes.

3. **Build transfer operators**

   - From the predicted \(B_h\), reconstruct the transfer operator \(Q\) and corresponding coarse‑level operators (via Galerkin projection).

4. **Use in a multigrid solver**

   - Insert the learned operators into a multigrid V‑cycle and compare:
     - Convergence rate
     - CPU time to assemble transfer operators
   - Compare against classical semi‑geometric multigrid based on exact (pseudo–\(L^2\) projection) operators.

For full algorithmic details, see Sections 2–5 of the paper.

---

## Reproducing the Paper’s Experiments

The scripts are designed to reproduce, at least partially, the numerical experiments reported in:

> C. Tomasi, R. Krause, *Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach*, arXiv:2109.05873.

Typical steps include:

1. Generate datasets for different mesh sizes / classes (as in Section 3 of the paper).
2. Train the network on 1D and 2D datasets.
3. Run multigrid experiments and collect:
   - Convergence plots
   - Timings for assembling the transfer operators
4. Compare with a baseline (semi‑geometric multigrid).

---

## License

This code is made available for **research and educational purposes**.

You are free to use, modify, and redistribute this code **provided that**:

1. You clearly acknowledge the original authors, and  
2. You **cite the associated paper** (see below) in any published work or derivative code that makes use of this repository.

If you need a different or more formal licensing scheme for your project, please contact the author.

---

## How to Cite

If you use this code or build on the ideas presented here, please cite the following work.

### arXiv version

```bibtex
@article{TomasiKrause2021NNMG,
  author  = {Tomasi, Claudio and Krause, Rolf},
  title   = {Construction of Grid Operators for Multilevel Solvers: a Neural Network Approach},
  journal = {arXiv preprint arXiv:2109.05873},
  year    = {2021},
  url     = {https://arxiv.org/abs/2109.05873}
}
```

### Springer / LNCS version (if preferred)

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
  pages     = {1--12},
  doi       = {10.1007/978-3-030-95025-5_63}
}
```

---

## Author

**Claudio Tomasi**  

(Repository author and paper main author)

If you have questions, find an issue, or would like to discuss extensions of this work, feel free to open an issue or contact the author.
