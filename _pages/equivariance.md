---
permalink: /equivariance/
title: "Research"
author_profile: true
redirect_from: 
  - /md/
  - /markdown.html
---
Notes and Toy Code: Equivariant Neural Networks
======
* Group CNNs and Steerable CNNs (used more frequently in computer vision/image data)
	* Lift the features to the group space and apply convolutions (weight sharing) in the group space to achieve group equivariance.
	* [Notes](https://wenhangao21.github.io/files/EquivariantNN_files/GroupCNN.pdf) 
	* [Toy Code](https://colab.research.google.com/drive/1b3QThdUuBhtOfZersKAR5WzhBKwGL3Db?usp=sharing) for a simple one channel implementation of group convolutions.

* Geometric GNNs (for point clouds, 3D molecular graphs, or any data with structural relationships)
	* Restrict the features to invariant ones (for invariant tasks only) or to equivariant ones with equivariance-preserving interactions.
	* [Notes](https://wenhangao21.github.io/files/EquivariantNN_files/Geometric_GNNs.pdf) 
	* [Toy Code](https://colab.research.google.com/drive/1pTNknyItKagFPB2SXUKV5SQxWNecFURH?usp=sharing) for a demonstration of decomposing catersian tensors into spherical tensors, equivariance, and irreducibility.
	
* Unconstrained Geometric GNNs (for point clouds, 3D molecular graphs, or any data with structural relationships)
	* Offset the equivariance to the data instead of the model; the model can be chosen freely (unconstrained).
	* [Notes](https://wenhangao21.github.io/files/EquivariantNN_files/EGNN_FA.pdf) 
	* [Toy Code](https://colab.research.google.com/drive/1kdTxFxAaObqgCvW_qThd7uTrHvKCvhAr?usp=sharing) for a demonstration of frame averaging for the O(3) group.
	

The materials will be regularly updated, with more content added over time. These notes contain a lot of my own understandings and intuitions; opinions are mine. If you find any errors/typos, please let me know!




