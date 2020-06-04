# dhsc: (Disordered) Heisenberg Spin chains

This repo was made for a course project in Numerical Algorithms for Scientific Computing (APC 523) at Princeton University during Spring 2018.

In this repo, I implement a matrix-product state algorithm to calculate approximate energy eigenvalues of a quantum system.
In particular, this technique is used to find the low energy states of a Heisenberg Spin Chain, a system of N particles, each with 2 degrees of freedom, coupled together by nearest-neighbor interactions.
Nominally, this is equivalent to solving the eigen decomposition problem for a matrix of size $$2^N \times 2^N$$, which is quickly intractable on classical supercomputers for $$N \geq 20$$.

$`\sqrt{2}`$

Matrix product states for Heisenberg spin chains


Thanks to Michael Muller.

edit (June 2020): added documentation, cleaned up some old files
