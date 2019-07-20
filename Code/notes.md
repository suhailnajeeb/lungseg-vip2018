### voxel dimensions

- Most of the scans have voxels of dimension (mm) : 0.97 x 0.97 x 3
- One has 0.812 x 0.812 x 3, and another has 0.898 x 0.898 x 3

### Evaluation Metrics
 
1. Dice Coefficient
2. Mean Surface Distance = average distance of a point in X to its closest point in Y
3. Hausfordf distance (95 %) = 95% Hausdorff distance is the point in X with distance to its closest point in Y is greater or equal to exactly 95% of the other points in X

- Hausdorff distance is the longest distance you can be forced to travel by an adversary who chooses a point in one of the two sets, from where you then must travel to the other set. In other words, it is the greatest of all the distances from a point in one set to the closest point in the other set.


### Scanning Machines
- In the Training set: 
  - 67 scans are from machine made by CMS Inc : scale of values : -1024 to 3071
  - 33 by SIEMENS : scale of values 0 to 4095
