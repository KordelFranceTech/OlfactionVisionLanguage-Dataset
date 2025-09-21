---
license: mit
task_categories:
- image-classification
- image-to-text
- robotics
language:
- en
tags:
- chemistry
- biology
- robotics
- olfaction
- olfactory
- smell
- scent
- odor
- odour
pretty_name: Olfaction-Vision-Language Dataset
size_categories:
- 10K<n<100K
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- expert-created
source_datasets:
- GoodScents
- LeffingWell
- COCO
---
Olfaction-Vision-Language Learning: A Multimodal Dataset
----

<div align="center">

**Olfaction • Vision • Language**


[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow?logo=google-colab)](https://colab.research.google.com/drive/1-VTEvfCZ3FC8PfxeynbLAWErYYkyNjfZ?usp=sharing)
[![Paper](https://img.shields.io/badge/Research-Paper-red)](https://arxiv.org/abs/2506.00455v3)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/datasets/kordelfrance/olfaction-vision-language-dataset)

</div>


An open-sourced dataset and dataset builder for olfaction-vision-language tasks.

---

## Dataset Description

- **Modalities**: Olfaction, Vision, Language

- **Data Format**:
  All sensor streams are synchronized and stored in a standardized JSON / NoSQL format.

- **Total Samples**: _~118,000_
- **Environments**: Indoor, outdoor, lab-controlled, and natural settings

---

## Getting Started

The easiest way to get started is to open the Colab notebook and begin there.
To explore the dataset locally, follow the steps below:

#### 1. Clone the Repository

```bash
git clone https://github.com/KordelFranceTech/OlfactionVisionLanguage-Dataset.git
cd OlfactionVisionLanguage-Data
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

#### 4. Run Sample Data Generation Notebook

```bash
jupyter notebook notebooks/Olfaction_Diffusion-Data-Generator.ipynb
```
---

## Limitations
While our integrated framework provides a foundational approach for
constructing olfaction-vision-language datasets
to enhancing robotic odour source localization, it is essential to
acknowledge its limitations and the challenges that persist in
scent-based navigation. VLMs, though powerful in bridging
visual and textual modalities, are not specifically trained on
olfactory datasets. Consequently, their ability to generate accu-
rate and comprehensive odour descriptors from images is con-
strained. This limitation can lead to incomplete or imprecise
odour representations, affecting the grounding of subsequent
molecular generation process. Moreover, VLMs may struggle
with contextual reasoning and spatial understanding, which are
crucial for accurately associating odours with their sources in
complex environments. This can be analogously observed from
other research (such as with Xie, et al. [1]) where they attempt to infer sound
from images using VLMs. Our dataset and generation method
inherit these limitations as a consequence. For example,
we noticed in our training that the VLM tends to associate
carbon monoxide to the presence of a vehicle in the COCO
image, but does not consider the fact that the vehicle may be
electric. However, due to the lack of robotics-centric olfaction
datasets, we emphasize that our methodology holds value.

Additionally, our method gives heavy credence to the Shape
Theory of Olfaction. If this theory is proven untrue, it may
invalidate the efficacy of our method. The generated molecules
require empirical validation to confirm their olfactory prop-
erties which can be accomplished through various olfaction
sensors. However, this can be very resource-intensive as the in-
strumentation required to validate the presence of compounds
is expensive. We note that Lee et, al. [2] also observed similar
attributes whose work of which we build on top. In addition,
obtaining all possible molecules over which to evaluate said
sensors can be restrictive due to regulations and required
licenses. Finally, even if one could obtain testing samples of all
possible compounds in a unanimous quantity, it is not enough
to test each compound individually. The combination and
interaction of certain compounds produce entirely new odour
descriptors which are not yet entirely predictable. 
Rapidly quantifying the presence of compounds within an air sample
and the aromas attributed to them is a known problem within
olfaction; this underscores the need for more datasets, learning
techniques, and grounding methods as proposed here.
Another significant challenge in olfactory navigation is the
accurate association of detected odours with their correct
sources. Environmental factors such as airflow dynamics, pres-
ence of multiple odour sources, and obstacles can cause odour
plumes to disperse unpredictably, leading to potential misattribution of odours to incorrect objects. While our framework
enhances the robot’s ability to infer and generate potential
odourant molecules, it does not eliminate the possibility of
such misassociations. Therefore, our system may still see
difficulties in environments with complex odour landscapes.
Implementing the proposed framework in real-time robotic
systems poses computational challenges. The integration of
VLMs, olfaction ML models, and olfactory sensors requires effi-
cient processing capabilities to ensure timely decision-making
during navigation. 

In summary of the above, we acknowledge that there are
inherent limitations of our proposed dataset, but hope
that it can be used to generate highly probable compounds for
given aromas when constructing vision-olfactory datasets and
informing sensor selection in olfactory robotics tasks.

For a more comprehensive overview of the limitations of this dataset 
and its construction, please refer to [the paper associated with this dataset](https://arxiv.org/abs/2506.00455v3).

References:

[1] Z. Xie, S. Yu, M. Li, Q. He, C. Chen, and Y.-G. Jiang, “Sonicvi-
sionlm: Playing sound with vision language models,” arXiv preprint
arXiv:2401.04394, 2024.

[2] B. K. Lee, E. J. Mayhew, B. Sanchez-Lengeling, J. N. Wei, W. W.
Qian, K. A. Little, M. Andres, B. B. Nguyen, T. Moloy, J. Yasonik,
J. K. Parker, R. C. Gerkin, J. D. Mainland, and A. B. Wiltschko,
“A principal odor map unifies diverse tasks in olfactory perception,”
Science, vol. 381, no. 6661, pp. 999–1006, 2023.


---

## Directory Structure

```text
OlfactionVisionLanguage-Dataset/
├── data/                     # Full and partial datasets
├── notebooks/                # Colab-ready notebooks
├── requirements.txt          # Python dependencies
├── LICENSE                   # Licensing terms of this repository
└── README.md                 # Overview of repository contributions and usage
```

---

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@misc{france2025diffusiongraphneuralnetworks,
      title={Diffusion Graph Neural Networks and Dataset for Robust Olfactory Navigation in Hazard Robotics}, 
      author={Kordel K. France and Ovidiu Daescu},
      year={2025},
      eprint={2506.00455},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00455v3}, 
}
```

---


## License

This dataset is released under the [MIT License](https://opensource.org/license/mit).
