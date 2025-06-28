## Repository 


## Dataset
https://github.com/SchedulingLab/fjsp-instances


### Brandimarte


### Hurink 
complexity increase with machine compatibility
- E-data (Easy): one job could process on few machine (average 1-1.15)
- R-data (Random): random 1~2 (average 2)
- V-data (Variable): (average 7.5)


### Format
First line: <number of jobs> <number of machines>
Then one line per job: <number of operations> and then, for each operation, <number of machines for this operation> and for each machine, a pair <machine> <processing time>.
Machine index starts at 0.

### Note
- Total Wighted Tardiness desined by Crauwels et al. (1998) ；OR-Library
    - https://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html

- $P=\sum_{j=1}^{n} p_j,\quad 
d_j \sim U\bigl[P(1-\text{TF}-\tfrac{\text{RDD}}2),\;P(1-\text{TF}+\tfrac{\text{RDD}}2)\bigr]$
