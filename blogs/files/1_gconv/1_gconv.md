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
  <img 
    alt="Symmetry Diagram"     src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/symmetry.png" 
    style="width: 50%; display: block; margin: 0 auto;" 
  />
  <br />
  <figcaption style="margin-top: 10px; text-align: center;">
    Figure 1: Symmetries in ML tasks and in nature.
  </figcaption>
</figure>

FYI: Dr. Chen Ning Yang from Stony Brook received the Nobel Prize in physics (1957) for discoveries about symmetries, and his B.S. thesis is “Group Theory and Molecular Spectra”.

### Learning Symmetries
To learn symmetry, a common approach is to do data-augmentation: Feed augmented data and hope the model “learns” the symmetry.

<figure style="text-align: center;">
  <img 
    alt="Data Augmentation"  src="https://raw.githubusercontent.com/wenhangao21/wenhangao21.github.io/refs/heads/main/blogs/files/1_gconv/data_augmentation.png" 
    style="width: 50%; display: block; margin: 0 auto;" 
  />
  <figcaption style="display: block; margin-top: 10px; text-align: center;">
    Figure 2: Data augmentation to learn symmetries.
  </figcaption>
</figure>

<span style="color: red;">Issues:</span>
- <span style="color: red;">No guarantee</span> of having symmetries in the model
- <span style="color: red;">Wasting valuable net capacity</span> on learning symmetries from data
- <span style="color: red;">Redundancy</span> in learned feature representation

<span style="color: green;">Solution:</span>
- Building symmetries into the model by design! 


