# Constraint-Driven AutoML

Optimizing a machine learning pipeline for a task at hand requires careful configuration of various hyperparameters, typically supported by an AutoML system that optimizes the hyperparameters for the given training dataset.
Yet, depending on the AutoML system's own second-order meta-configuration, the performance of the AutoML process can vary significantly. Current AutoML systems cannot automatically adapt their own configuration to a specific use case. Further, they cannot compile user-defined application constraints on the effectiveness and efficiency of the pipeline and its generation.
In this paper, we propose CAML, which uses meta-learning to automatically adapt its own AutoML parameters, such as the search strategy, the validation strategy, and the search space, for a task at hand. The dynamic AutoML strategy of CAML takes user-defined constraints into account and obtains constraint-satisfying pipelines with high predictive performance. 

![image](https://user-images.githubusercontent.com/5217389/216223724-05dd746d-4cce-4e64-869e-b791cfe7cee2.png)

Credit Stable Diffusion

Check out the example of CAML here:
https://colab.research.google.com/drive/1z8EnNQnik5gNKSTf38YIDGk4ivJlRVpi?usp=sharing


## Setup
```
conda create -n AutoMLD python=3.7
conda activate AutoMLD
cd Software/DeclarativeAutoML/
git pull origin main
sh setup.sh
```

## CAML in four lines of code

```python
from fastsklearnfeature.declarative_automl.caml.CAML import CAML
caml = CAML(time_search_budget=60, inference_time_limit=0.001)
caml.fit(X_train, y_train, categorical_indicator)
y_pred = caml.predict(X_test)
```

## Relevant publication

If you use CAML in scientific publications, we would appreciate citations.

**AutoML in Heavily Constrained Applications**
*Felix Neutatz, Marius Lindauer, and Ziawasch Abedjan*
VLDB Journal (2023)

[Link](https://arxiv.org/pdf/2306.16913.pdf) to publication.
```
@inproceedings{caml,
    title     = {AutoML in Heavily Constrained Applications},
    author    = {Felix Neutatz and
                  Marius Lindauer and
                  Ziawasch Abedjan},
    booktitle = {VLDB Journal},
    year      = {2023}
}
```
