09/01/2019
Copyright@Jiamian Hu's group
Contact: Minyi Dai (mdai26@wisc.edu)
This file inroduces detailed information of the datasets.
------------------------------------------------------------------

The datasets now have 350 data points. The information of each data point are all included in the corresponding subfolder. 
In each subfolder, there are three files:

1.feature.txt: containing information of each grain/feature in the 3D microstructure.

Column1: Grain/Feature Number 
Column2: Euler angle 1 (radians)
Column3: Euler angle 2 (radians)
Column4: Euler angle 3 (radians)
Column5: Grain size (grid)
Column6: Number of Neighbors.

Number of grains(rows) is varied for different feature.txt and is determined based on the microstructure.

2.neighbor.txt: containing a large matrix to decribe whether two grains are neighbors or not.

           Grain #1  Grain #2  Grain #3 ...     
Grain #1      0			0		  1		    		  
Grain #2      0         0         0         
Grain #3      1         0         0
   .
   .
   .
   
0 means two grains are not neighbors and 1 means two grains are neighbors. For example, grain #1 and grain #2 are not neighbors while grain #1 and grain #3 are neighbors.
Obviously, this matrix is symmetric. Also, all diagonal elements is zero.

3. Property.txt : containing information of magnetostrictive response. 
Column1: applied magnetic field (kA/m)
Column2: strain (unitless)

For propert.txt in subfolder from structure-1 to structure-100, it has 4 rows. For file in other subfolder, it has 5 rows.