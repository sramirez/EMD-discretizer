# EMD-discretizer

## Description:

Here we present a new evolutionary-based discretization algorithm called EMD (also ECPRSD) which selects the most adequate combination of boundary cut points to create discrete intervals. For this purpose, EMD uses a wrapper fitness function based on the classification error provided by two important classifiers, and the number of cut points produced. The proposed algorithm follows a multivariate approach, being able to take advantage of the existing interactions and dependencies among the set of input attributes and the class output to improve the discretization process. It also includes a chromosome reduction mechanism to tackle larger problems and, in general, to speed up its performance on all kinds of datasets.

## Installation:

The source code uploaded is designed to be integrated into the KEEL software tool: http://www.keel.es/

## Associated paper:

S. Ramírez-Gallego, S. García, J. M. Benítez and F. Herrera, "Multivariate Discretization Based on Evolutionary Cut Points Selection for Classification," in IEEE Transactions on Cybernetics, vol. 46, no. 3, pp. 595-608, March 2016.
doi: 10.1109/TCYB.2015.2410143
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7063251&isnumber=7406788

