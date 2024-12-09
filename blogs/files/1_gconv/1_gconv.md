---
layout: blog
title: "Technical Blogs"
author_profile: false
---
<div style="height: 20px;"></div>

# Group Convolution Neural Networks

## Introduction

### Why Symmetries
Group equivariance in ML models is about enforcing symmetries in the architectures.
- Many learning tasks, oftentimes, have symmetries under some set of transformations acting on the data.
	- For example, in image classification, rotating or flipping an image of a cat should not change its classification as a "cat."
- More importantly, the nature itself is about symmetries.
	- Similar symmetries appear in physical systems, molecular structures, and many other scientific data.
	
<figure style="text-align: center;">
  <img alt="Symmetry Diagram" width="50%" src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/symmetry.png" />
  <figcaption>Figure 1: Symmetries in ML tasks and in nature.</figcaption>
</figure>

FYI: Dr. Chen Ning Yang from Stony Brook received the Nobel Prize in physics (1957) for discoveries about symmetries, and his B.S. thesis is “Group Theory and Molecular Spectra”.
