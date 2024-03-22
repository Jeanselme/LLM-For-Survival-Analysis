# How to use Large Language Models for Survival Analysis ?

This code is the implementation of the framework described in [Review and Evaluation of Language Models for Survival Analysis](https://openreview.net/forum?id=ZLUsZ52ibx) presented at the AAAI Clinical Foundation Models Workshop.

Our work identified different approaches to use LLMs and adapt them for survival analysis (Figure 1). Due to the lack of recommendation, we introduced this framework to create novel guidelines.

![LLM For Survival Analysis](img/sallm.jpg?raw=true "Figure 1: Survival analysis from unstrucutred data")

### How to contribute?
#### Compare on your dataset
If you have available data, *data that has not been used to train the different LLMs*, apply the different strategies to improve guidelines. Please share your results by updating the `evidence.csv` following the proposed template.

#### Improve the codebase
Our approach compares a limited set of publicly available LLMS. As methods are continously release, please help us integrating new strategies to improve this comparison.

### How to run ?
The notebooks aim to provide a step by step tutorial to use on your dataset.
1. Run `0. Preprocessing.ipynb` to preprocess the TCGA dataset (this script needs to be adapted to your own dataset).
2. Run `1. Embedding.ipynb` to extract the emebedding of all unstrucutred data (this allows to avoid rerunning the extraction).
3. Run `LLM1. Extract Concept.ipynb` for automatic extraction of the structured data from the unstructured one (using the different LLM strategies presented in Figure 1).
4. Run `LLM2. Predictions.ipynb` for predictions from the unstructured data.
5. Analyse the results with  `LLM. Analysis.ipynb` which presents a survival evalaution.

6. If you run your analysis on a new dataset, please create a pull request with an updated `evidence.csv`. 

### Setup
#### Structure
All scripts are at the root, with LLMHit (an extension of DeepHit using language model) included in `model/`.

#### Clone
```
git clone git@github.com:Jeanselme/LLM-For-Survival-Analysis.git
```

#### Requirements
The model relies on `pytorch`, `pandas`, `pycox`, `numpy`, `scikit-learn`, `transformers`.  

#### Reference
Please reference the following if using this code:
```
@inproceedings{jeanselme2024language,
  title={Review and Evaluation of Language Models for Survival Analysis},
  author={Jeanselme, Vincent and Agarwal, Nikita and Wang, Chen},
  booktitle={AAAI 2024 Spring Symposium on Clinical Foundation Models},
  year={2024}
}
```