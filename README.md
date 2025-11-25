# LearnMultigrid
Reference implementation of the multigrid method described in the paper:

> **C. Tomasi**  
> *A Simple Multigrid Implementation for Educational and Research Purposes*  
> In: Fidanova S. (eds) Advances in High Performance Computing.  
> ISC High Performance 2021.  
> Lecture Notes in Computer Science, vol 13209.  
> Springer, Cham.  
> https://doi.org/10.1007/978-3-030-95025-5_63

This repository contains the full source code (archived as a ZIP file) accompanying the research chapter above.

---

## Overview

**LearnMultigrid** is a self-contained C++ project implementing a clear and didactic version of a geometric multigrid solver.

---

## Contents of the ZIP

```
LearnMultigrid/
├── src/
├── include/
├── examples/
├── grids/
├── cycles/
├── solvers/
├── CMakeLists.txt
└── README.md
```

---

## Features
- Geometric multigrid solver  
- Jacobi, weighted Jacobi, Gauss–Seidel smoothers  
- Restriction & prolongation operators  
- V-cycle implementation  
- Didactic code structure  

---

## Building

### Requirements
- C++17 compiler  
- CMake ≥ 3.10  

### Build
```bash
mkdir build
cd build
cmake ..
make
```

---

## Related Publication  
https://link.springer.com/chapter/10.1007/978-3-030-95025-5_63

---

## How to Cite

### BibTeX
```bibtex
@incollection{Tomasi2021Multigrid,
  author    = {Tomasi, Claudio},
  title     = {A Simple Multigrid Implementation for Educational and Research Purposes},
  booktitle = {Advances in High Performance Computing},
  editor    = {Fidanova, Stefka},
  year      = {2021},
  publisher = {Springer},
  address   = {Cham},
  pages     = {745--756},
  doi       = {10.1007/978-3-030-95025-5_63}
}
```

---

## License
This work is released under a **citation-required license**:  
You are free to use, modify, and redistribute the code **as long as you cite the paper above**.

---

## Author
Claudio Tomasi
