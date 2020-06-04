# dhsc: (Disordered) Heisenberg Spin chains

This repo was made for a course project in Numerical Algorithms for Scientific Computing (APC 523) at Princeton University during Spring 2018.

In this repo, I implement a matrix-product state algorithm to calculate approximate energy eigenvalues of a quantum system.
In particular, this technique is used to find the low energy states of a Heisenberg Spin Chain, a system of N particles, each with 2 degrees of freedom, coupled together by nearest-neighbor interactions.
Nominally, this is equivalent to solving the eigen decomposition problem for a matrix of size $2^N \times 2^N$, which is quickly intractable on classical supercomputers for $N \geq 20$.
However, by exploiting the sparsity and locality of the system, one can use a *tensor-network* based method to compress this system using a number of parameters that scales *polynomially*, not *exponentially* with $N$.

The upshot: One can solve systems with $N = 64$, to very high accuracy (over 99.99999% accuracy compared to the exact Bethe ansatz result), in a matter of seconds on a laptop. Nominally, this state is a vector of dimension $2^{64} \approx 1.8 \times 10^{19}$.


Read the report (Report.pdf) for more details, or see the other Jupyter notebook (demo.ipynb) for a demo.



Thanks to Michael Muller, who taught the course.

edits (June 2020): 
 - changed repo visibility from private to public
 - added documentation
 - cleaned up some old files



```python

```
