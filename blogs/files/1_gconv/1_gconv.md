---
layout: blog
title: "Technical Blogs"
author_profile: false
---

# Group Convolution Neural Networks

## Introduction: Why Symmetries
Group equivariance in ML models is about enforcing symmetries in the architectures.
- Many learning tasks, oftentimes, have symmetries under some set of transformations acting on the data.
	- For example, in image classification, rotating or flipping an image of a cat should not change its classification as a "cat."
- More importantly, the nature itself is about symmetries.
	- Similar symmetries appear in physical systems, molecular structures, and many other scientific data.
	
<img align="right" alt="png" width="80%" src="symmetry.png" />