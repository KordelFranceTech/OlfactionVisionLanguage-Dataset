# Olfaction-Vision-Language Learning: Diffusion-Graph Models with Dataset
----

<div align="center">

**Vision • Olfaction • Language**


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow?logo=google-colab)](https://colab.research.google.com/drive/1-VTEvfCZ3FC8PfxeynbLAWErYYkyNjfZ?usp=sharing)
[![Paper](https://img.shields.io/badge/Research-Paper-red)](https://arxiv.org/abs/2506.00455)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces)

</div>


An open-sourced dataset, dataset builder, and diffusion model for olfaction-vision-language tasks.

---

## Dataset Description

- **Modalities**:
  - Olfactory (based on the GoodScent and Leffingwell datasets)
  - Visual (based on the COCO dataset)
  - Language (LLM-generated descriptors)

- **Data Format**:
  All sensor streams are synchronized and stored in a standardized JSON / NoSQL format.

- **Total Samples**: _~5,000_
- **Environments**: Indoor, outdoor, lab-controlled, and natural settings

---

## Running Locally

To explore the dataset locally, follow the steps below:

#### 1. Clone the Repository

```bash
git clone https://github.com/KordelFranceTech/DiffusionGraphOlfactionModels.git
cd DiffusionGraphOlfactionModels
````

#### 2. Create a Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run Sample Exploration Script

```bash
python scripts/visualize_data.py
```

---

## Directory Structure

```text
MultimodalDataset/
├── data/                     # Example data samples
├── scripts/                  # Data loading and visualization tools
├── notebooks/                # Colab-ready notebooks
├── models/                   # Pre-trained models for immediate use
├── requirements.txt          # Python dependencies
├── LICENSE                   # Licensing terms of this repository
└── README.md                 # Overview of repository contributions and usage
```

---

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@misc{france2025diffusiongraphneuralnetworks,
      title={Diffusion Graph Neural Networks for Robustness in Olfaction Sensors and Datasets}, 
      author={Kordel K. France and Ovidiu Daescu},
      year={2025},
      eprint={2506.00455},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00455}, 
}
```

---


## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

