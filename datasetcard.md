---
# Olfaction-Vision-Language Dataset Card  
language:
- en
license: mit
tags:
- olfaction
- vision
- language
- multimodal
- chemistry
- robotics
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- expert-created
language_details:
- en-US
pretty_name: OVLM
size_categories:
- 5000
source_datasets:
- GoodScents
- LeffingWell
- COCO
task_categories:
- image-classification
- image-to-text
- robotics
- other
task_ids:
- multi-label-image-classification
- image-captioning
- task-planning
- olfaction


extra_gated_fields:
- Name: text  # Example: Name: text
- Affiliation: text  # Example: Affiliation: text
- Email: text  # Example: Email: text
- I understand that this dataset is an experimental dataset generated for multimodal robotics and deep learning research, and that this dataset is provided as is. I understand that this dataset should not be used to make any claims in medical or healthcare applications.: checkbox  # Example for speech datasets: I agree to not attempt to determine the identity of speakers in this dataset: checkbox
extra_gated_prompt: By clicking on “Access repository” below, you also agree to not attempt to the conditions checked above.
