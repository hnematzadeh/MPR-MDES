# MPR-MDES

This is the implementation of a paper entitled "Pattern recognition frequency-based feature selection with multi-objective discrete
evolution strategy on high dimentional medical datasets". The proposed feature selection method (MPR-MDES) is a hybrid (filter + wrapper) method which is a direct furture work of [Automatic frequency-based feature selection using discrete weighted evolution strategy](https://https://www.sciencedirect.com/science/article/pii/S1568494622007487#!).

Maximum Pattern Recognition (MPR) is a frequency-based filter ranking method, which belongs to a series of frequency-based rankers:

1- [Mutual Congestion (MC)](https://www.sciencedirect.com/science/article/pii/S0888754318304245)   Publication year: 2019

2- [Sorted Label Interference (SLI)](https://www.sciencedirect.com/science/article/pii/S0306437921000259#!)   Publication year: 2021

3- [Sorted Label Interference-gamma (SLI-gamma)](https://link.springer.com/article/10.1007/s11227-022-04650-w)   Publication year: 2022

4- [Extended Mutual Congestion (EMC)](https://https://www.sciencedirect.com/science/article/pii/S1568494622007487#!).  Publication year: 2022

##################### Instruction #########################

After loading the corresponding dataset in your local drive:


1- Run lines 55-71 to calculate the summation of samples per label

2- Run lines 73-87 for Maximum Pattern Recognition (MPR)

3- Run lines 90-104 to create a dataset with top 20 features of MPR

4- Run lines 109-403 for Multi-objective Discrete Evolution Strategy and its corresponding functions
 
