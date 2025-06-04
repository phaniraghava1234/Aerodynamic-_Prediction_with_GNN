# Aerodynamic Flow Prediction with Physics-Informed Graph Neural Networks

## Project Overview

This project explores the application of Geometric Deep Learning (GDL), specifically Graph Neural Networks (GNNs), for accelerating aerodynamic flow prediction. Leveraging the high-fidelity **AirfRANS dataset**, this work develops a GNN-based surrogate model capable of predicting complex flow fields (velocity, pressure, turbulent kinematic viscosity) around 2D airfoils. A key focus is the integration of **physics-informed features** to enhance model accuracy, generalization, and physical consistency, alongside a preliminary exploration of **uncertainty quantification**.

The traditional reliance on computationally expensive Computational Fluid Dynamics (CFD) simulations often bottlenecks aerodynamic design cycles. This project demonstrates how GNNs can provide rapid, high-resolution predictions, offering a significant step towards efficient, data-driven aerospace design and analysis.

## Key Features & Components

* **Graph Neural Network (GNN) Model:** Implementation of a GNN architecture (e.g., PointNet++ inspired or Message Passing based) designed to process unstructured point cloud data from CFD simulations.
* **AirfRANS Dataset Integration:** Utilizes the `torch_geometric.datasets.AirfRANS` dataset, handling its point cloud structure and constructing appropriate graph representations (e.g., using `KNNGraph` or `RadiusGraph`).
* **Physics-Informed Feature Engineering:** Incorporation of domain-specific knowledge by deriving and integrating features such as approximate local Reynolds number ($Re_x$) and conceptually, inviscid pressure distribution ($c_{p,inviscid}$), to guide the GNN towards physically consistent predictions.
* **Flow Field Prediction:** Node-level regression to predict multiple aerodynamic quantities (velocity components, pressure, turbulent kinematic viscosity) across the airfoil and surrounding flow domain.
* **Uncertainty Quantification (Preliminary):** Exploration of basic uncertainty estimation techniques (e.g., Monte Carlo Dropout) to provide confidence bounds for predictions, crucial for high-stakes engineering applications.
* **Performance Analysis & Visualization:** Comprehensive evaluation using quantitative metrics (MSE, R-squared) and qualitative visualizations (pressure contours, $C_p$ plots) to assess model accuracy and generalization.

## Technical Stack

* **Python:** Core programming language.
* **PyTorch:** Deep learning framework.
* **PyTorch Geometric (PyG):** Library for implementing GNNs and handling graph data.
* **NumPy:** Numerical computing.
* **Matplotlib / Seaborn:** Data visualization.
* **Scikit-learn:** Utility functions (e.g., data scaling, metrics).
* **XFOIL (External):** Tool for generating inviscid pressure data (used for feature engineering, potentially for a subset of data).

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)  # For CUDA 11.8, adjust the CUDA version if needed or use 'cpu' if you don't have a GPU
    pip install torch_geometric
    pip install numpy matplotlib scikit-learn
    ```
4.  **Download AirfRANS Dataset:** The dataset will be automatically downloaded by PyTorch Geometric when you first access it in your code.

## Usage

The project is structured into several Python scripts, each focusing on a different aspect of the GNN development.

* `data_preprocessing.py`: Handles loading the AirfRANS dataset, applying graph transforms (KNNGraph/RadiusGraph), and normalizing features.
* `model_architecture.py`: Defines the GNN model (e.g., PointNet++ inspired).
* `train.py`: Contains the training and validation loop, including loss function, optimizer, and basic evaluation.
* `feature_engineering.py`: (To be implemented/expanded) Script for calculating and integrating physics-informed features.
* `evaluate.py`: Performs comprehensive evaluation and generates visualizations.

To run the full pipeline:


# Example: Run the training script
python train.py
# Example: Run evaluation after training
python evaluate.py

# References

Here is a list of references, formatted in a common scientific paper style, for the resources and concepts utilized throughout this project's planning and development:

1.  **AirfRANS Dataset Documentation:** The official documentation for the AirfRANS dataset, detailing its structure, contents, and usage.
    * [https://airfrans.readthedocs.io/en/latest/](https://airfrans.readthedocs.io/en/latest/)
    * [https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS)

2.  **Jena, T., Morvan, S., & Benard, N. (2025). Predicting airfoil pressure distribution using boundary graph neural networks.** *arXiv preprint arXiv:2503.18638*.
    * [https://arxiv.org/abs/2503.18638](https://arxiv.org/abs/2503.18638)
    * *Note: This paper was foundational for the physics-informed features and B-GNN concepts discussed in this project.*

3.  **H. Wu, P. J. & B. L. (2024). DrivAerNet++: A Large-Scale Multi-Modal Dataset for Aerodynamic Car Design.** *arXiv preprint arXiv:2402.13840*.
    * [https://arxiv.org/abs/2402.13840](https://arxiv.org/abs/2402.13840)
    * *Associated GitHub Repository:* [https://github.com/Mohamedelrefaie/DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet)

4.  **Wang, W., Li, J., & Cai, C. (2023). EAGLE: A Large-Scale Dataset for Unsteady Fluid Dynamics.** *NeurIPS 2023 Datasets and Benchmarks Track*.
    * [https://eagle-dataset.github.io/](https://eagle-dataset.github.io/)

5.  **Perić, D., Jasa, J., Vrhovac, P., & Milovanović, V. (2024). WindsorML: A High-Fidelity CFD Dataset for Machine Learning in Automotive Aerodynamics.** *NeurIPS 2024 Datasets and Benchmarks Track*.
    * [https://proceedings.neurips.cc/paper_files/paper/2024/file/42a59a5f35b1b3c3fd648397c88a7164-Supplemental-Datasets_and_Benchmarks_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/42a59a5f35b1b3c3fd648397c88a7164-Supplemental-Datasets_and_Benchmarks_Track.pdf)
    * *Associated S3 Bucket:* `s3://caemldatasets/windsor/dataset`

6.  **NASA Common Research Model Website:** A collection of reference geometries and associated computational/experimental data for aeronautical research.
    * [https://commonresearchmodel.larc.nasa.gov/](https://commonresearchmodel.larc.nasa.gov/)

7.  **Jain, A., & Gupta, A. (2021). Learning to Simulate with Graph Neural Networks.** *NVIDIA Developer Blog*.
    * [https://developer.nvidia.com/blog/learning-to-simulate-with-graph-neural-networks/](https://developer.nvidia.com/blog/learning-to-simulate-with-graph-neural-networks/)
    * *Associated MeshGraphNet (MGN) implementation:* [https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/aero_graph_net](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/external_aerodynamics/aero_graph_net)

8.  **Wang, H., & Chen, G. (2021). Multi-Grid Graph Neural Networks with Self-Attention for Fluid Simulations.** *arXiv preprint arXiv:2104.09033*.
    * [https://arxiv.org/abs/2104.09033](https://arxiv.org/abs/2104.09033)

9.  **Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.** *Advances in Neural Information Processing Systems, 30*.
    * [https://arxiv.org/abs/1706.02413](https://arxiv.org/abs/1706.02413)

10. **PyTorch Geometric Documentation:** Comprehensive documentation for various modules, layers, and datasets within the PyTorch Geometric library.
    * [https://pytorch-geometric.readthedocs.io/en/latest/](https://pytorch-geometric.readthedocs.io/en/latest/)

11. **Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research, 12*, 2825-2830.
    * [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

12. **Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.** *International Conference on Machine Learning*.
    * [https://arxiv.org/abs/1506.02142](https://arxiv.org/abs/1506.02142)

13. **Vazquez, D., & Benard, N. (2023). A Multi-Fidelity Graph U-Net Model for Accelerated Physics Simulations.** *arXiv preprint arXiv:2307.16546*.
    * [https://arxiv.org/abs/2307.16546](https://arxiv.org/abs/2307.16546)

14. **You, Y., et al. (2021). Uncertainty Quantification over Graph with Conformalized Graph Neural Networks.** *NeurIPS 2021*.
    * [https://proceedings.neurips.cc/paper_files/paper/2021/file/71c4ac472e391307682f6f43e3713600-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/71c4ac472e391307682f6f43e3713600-Paper.pdf)