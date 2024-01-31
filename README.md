# Ablation Studies in Activation Maps for Explainable Semantic Segmentation in Industry 4.0

This repository contains the implementation of the Ablation Studies in Activation Maps for Explainable Semantic Segmentation in Industry 4.0, as presented at IEEE Eurocon 2023.
In particular we provided an extension of Ablation-CAM to semantic segmentation.
The research paper associated with this implementation can be found in the proceedings of IEEE Eurocon 2023. You can access the paper [here](https://ieeexplore.ieee.org/abstract/document/10199094).

## Abstract

In recent years, much attention has been given to various eXplainable Artificial Intelligence (XAI) and interpretability methods. Their extension to dense prediction tasks, however, has been underexplored. Gradient-based saliency maps, highlighting feature importance in terms of input pixels, have been frequently used as fast and simple visual explanation techniques. Nonetheless, they face several problems, and the exploration of different types of attribution methods is warranted. In this paper, we investigate gradient-free semantic segmentation explanations that are based on ablating activation maps. We explore their potential for industrial applications, specifically for fruit pitting machines. We also extend the application of Ablation-CAM, a gradient-free ablation-based interpretability technique, to semantic segmentation. Finally, we discuss the sensitivity of activation maps to partial occlusions of either the foreground or the background class regions.

## Installation

Check the environment.ylm file.

## Usage
By running the code, it is possible to:

- Plot the output of Ablation-CAM for semantic segmentation when applied on the last encoder layer.
- Display the resized and smoothed Ablation-CAM output.
- Overlay the Ablation-CAM output on the original input image.

Additionally, by modifying the code, it is possible to access XAI for other layers, providing flexibility in exploring different levels of interpretability.


## Acknowledgment
This work was supported by the following projects:

- 4I: mixed reality, machine learning, gamification and educational for Industry”, Prog. n. F19013001-03X44, PON I&C 2014-2020, CUP: B66G21000040005 COR:4641138.
- PNRR project FAIR - Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
- PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN 00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU.
