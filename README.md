## Introduction
We propose a model, **SERMON** (Aspect-enhanced Explainable Recommendation with Multi-modal Contrast Learning), to explore the application of multimodal contrastive learning to facilitate reciprocal learning across two modalities and thereby improve the modeling of user preferences. 
## Usage
Below are examples of how to run SERMON.
```
python -u main.py \
--data_path ./reviews.json \
--cuda \
--checkpoint ./tripadvisorf/ \
```
## Code dependencies
-   Python== 3.7
-	PyTorch ==1.12.1
-	transformers==4.25.1
-	pandas==1.4.3
-	mkl-service==2.4.0
-	nltk==3.7
-	tokenizers==0.13.2
-	ply==3.11
## Files 
- main.py is used for train a SERMON model.
- model.py is the construction and details of the model.
- utils.py has functions for processing input files.
- prediction.py some implementation of prediction layer 
- bleu.py and rouge.py folder contains the tool and a example script of text evaluation.