# Interpretable Machine Learning with Python

### Overview

Usage of AI and machine learning models is likely to become more commonplace as larger swaths of the economy embrace automation and data-driven decision-making. While these predictive systems can be quite accurate, they are usually treated as inscrutable black boxes that produce only numeric predictions with no accompanying explanations. Unfortunately, recent studies and recent events have drawn attention to mathematical and sociological flaws in prominent weak AI and ML systems, but practitioners don’t often have the right tools to pry open machine learning models and debug them. This Oriole is part of a series that introduces several new approaches to that increase transparency, accountability, and trustworthiness in machine learning models. If you are a data scientist or analyst and you want to explain a machine learning model to your customers or managers (or if you have concerns about documentation, validation, or regulatory requirements), then this series of Jupyter notebooks is for you!

The notebooks highlight techniques such as:
* [Monotonic XGBoost models, partial dependence, and individual conditional expectation plots](https://github.com/jphall663/interpretable_machine_learning_with_python#enhancing-transparency-in-machine-learning-models-with-python-and-xgboost---notebook)
* [Decision tree surrogates, reason codes, and ensembles of explanations](https://github.com/jphall663/interpretable_machine_learning_with_python#increase-transparency-and-accountability-in-your-machine-learning-project-with-python---notebook)
* [LIME](https://github.com/jphall663/interpretable_machine_learning_with_python#explain-your-predictive-models-to-business-stakeholders-with-lime-using-python-and-h2o---notebook)
* [Sensitivity and residual analysis](https://github.com/jphall663/interpretable_machine_learning_with_python#testing-machine-learning-models-for-accuracy-trustworthiness-and-stability-with-python-and-h2o---notebook)

The notebooks can be accessed through:
* [O'Reilly Safari](https://github.com/jphall663/interpretable_machine_learning_with_python#oreilly-safari-recommended)
* [Docker container](https://github.com/jphall663/interpretable_machine_learning_with_python#docker-installation-recommended)
* [Manual installation](https://github.com/jphall663/interpretable_machine_learning_with_python#manual-installation)

***

### Enhancing Transparency in Machine Learning Models with Python and XGBoost - [Notebook](https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/xgboost_pdp_ice.ipynb)

![](./readme_pics/pdp_ice.png)

Monotonicity constraints can turn opaque, complex models into transparent, and potentially regulator-approved models, by ensuring predictions only increase or only decrease for any change in a given input variable. In this notebook, I will demonstrate how to use monotonicity constraints in the popular open source gradient boosting package XGBoost to train a simple, accurate, nonlinear classifier on the UCI credit card default data.

Once we have trained a monotonic XGBoost model, we will use partial dependence plots and ICE plots to investigate the internal mechanisms of the model and to verify its monotonic behavior. Partial dependence plots show us the way machine-learned response functions change based on the values of one or two input variables of interest, while averaging out the effects of all other input variables. ICE plots can be used to create more localized descriptions of model predictions, and ICE plots pair nicely with partial dependence plots.


### Increase Transparency and Accountability in Your Machine Learning Project with Python - [Notebook](https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/dt_surrogate_loco.ipynb)

![](./readme_pics/dt_surrogate.png)

Gradient boosting machines (GBMs) and other complex machine learning models are popular and accurate prediction tools, but they can be difficult to interpret. Surrogate models, feature importance, and reason codes can be used to explain and increase transparency in machine learning models. In this Oriole, we will train a GBM on the UCI credit card default data. Then we’ll train a decision tree surrogate model on the original inputs and predictions of the complex GBM model and see how the variable importance and interactions displayed in the surrogate model yield an overall, approximate flowchart of the complex model’s predictions. We will also analyze the global variable importance of the GBM and compare this information to the surrogate model, to our domain expertise, and to our reasonable expectations.

To get a better picture of the complex model’s local behavior and to enhance the accountability of the model’s predictions, we will use a variant of the leave-one-covariate-out (LOCO) technique. LOCO enables us to calculate the local contribution each input variable makes toward each model prediction. We will then rank the local contributions to generate reason codes that describe, in plain English, the model’s decision process for every prediction.


### Explain Your Predictive Models to Business Stakeholders with LIME using Python and H2O - [Notebook](https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/lime.ipynb)

![](./readme_pics/lime.png)

Machine learning can create very accurate predictive models, but these models can be almost impossible to explain to your boss, your customers, or even your regulators. This Oriole will use LIME to increase transparency and accountability in a complex GBM model trained on the UCI credit card default data. LIME is a method for building linear surrogate models for local regions in a data set, often single rows of data. LIME sheds light on how model predictions are made and describes local model mechanisms for specific rows of data. Because the LIME sampling process may feel abstract to some practitioners, this Oriole will also introduce a more straightforward method of creating local samples for LIME.

Once local samples have been generated, we will fit LIME models to understand local trends in the complex model’s predictions. LIME can also tell us the local contribution of each input variable toward each model prediction, and these contributions can be sorted to create reason codes -- plain English explanations of every model prediction. We will also validate the fit of the LIME model to enhance trust in our explanations using the local model’s R2 statistic and a ranked predictions plot.

### Testing Machine Learning Models for Accuracy, Trustworthiness, and Stability with Python and H2O - [Notebook](https://github.com/jphall663/interpretable_machine_learning_with_python/blob/master/resid_sens_analysis.ipynb)

![](./readme_pics/resid.png)

Because machine learning model predictions can vary drastically for small changes in input variable values, especially outside of training input domains, sensitivity analysis is perhaps the most important validation technique for increasing trust in machine learning model predictions.
Sensitivity analysis investigates whether model behavior and outputs remain stable when input data is intentionally perturbed, or other changes are simulated in input data. In this Oriole, we will enhance trust in a complex credit default model by testing and debugging its predictions with sensitivity analysis.

We’ll further enhance trust in our model using residual analysis. Residuals refer to the difference between the recorded value of a target variable and the predicted value of a target variable for each row in a data set. Generally, the residuals of a well-fit model should be randomly distributed, because good models will account for most phenomena in a data set, except for random error. In this Oriole, we will create residual plots for a complex model to debug any accuracy problems arising from underfitting or outliers.

## Using the Examples

### O'Reilly Safari (recommended)

The ideal way to use these notebook is through [O'Reilly Safari](https://www.safaribooksonline.com/). Doing so will enable video narration by the notebook author and no installation of software packages is required. Individual lessons can be accessed below on Safari.

* [VIDEO: Monotonic XGBoost models, partial dependence, and individual conditional expectation plots](https://www.safaribooksonline.com/oriole/enhancing-transparency-in-machine-learning-models-with-python-and-xgboost)
* [VIDEO: Decision tree surrogates, reason codes, and ensembles of explanations](https://www.safaribooksonline.com/oriole/increase-transparency-and-accountability-in-your-machine-learning-project-with-python)
* [VIDEO: LIME](https://www.safaribooksonline.com/oriole/explain-your-predictive-models-to-business-stakeholders-w-lime-python-h2o)
* [VIDEO: Sensitivity and residual analysis](https://www.safaribooksonline.com/oriole/testing-ml-models-for-accuracy-trustworthiness-stability-with-python-and-h2o)

To use these notebooks outside of the Safari platform, follow the instructions below.

### Docker Installation (recommended)

A Dockerfile is provided to build a docker container with all necessary packages and dependencies. This is the easiest way to use these examples if you are on Mac OS X, \*nix, or Windows 10. To do so:

1. Install and start [docker](https://www.docker.com/).

From a terminal:

2. Create a directory for the Dockerfile.</br>
`$ mkdir anaconda_py35_h2o_xgboost_graphviz`

3. Fetch the Dockerfile from the mli-resources repo.</br>
`$ curl https://raw.githubusercontent.com/jphall663/interpretable_machine_learning_with_python/master/anaconda_py35_h2o_xgboost_graphviz/Dockerfile > anaconda_py35_h2o_xgboost_graphviz/Dockerfile`

4. Build a docker image from the Dockefile.</br>
`$ docker build anaconda_py35_h2o_xgboost_graphviz`

5. Display docker image IDs. You are probably interested in the most recently created image. </br>
`$ docker images`

6. Start the docker image and the Jupyter notebook server.</br>
 `$ docker run -i -t -p 8888:8888 <image_id> /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/mli-resources --ip='*' --port=8888 --no-browser"`

7. Navigate to port 8888 on your machine.


### Manual Installation

1. Anaconda Python 4.2.0 from the [Anaconda archives](https://repo.continuum.io/archive/).
2. [Java](https://java.com/download).
3. The latest stable [h2o](https://www.h2o.ai/download/) Python package.
4. [Git](https://git-scm.com/downloads).
5. [XGBoost](https://github.com/dmlc/xgboost) with Python bindings.
6. [GraphViz](http://www.graphviz.org/).

Anaconda Python, Java, Git, and GraphViz must be added to your system path.

From a terminal:

7. Clone the mli-resources repository with examples.</br>
`$ git clone https://github.com/jphall663/interpretable_machine_learning_with_python.git`

8. `$ cd interpretable_machine_learning_with_python`

9. Start the Jupyter notebook server.</br>
`$ jupyter notebook`

10. Navigate to the port Jupyter directs you to on your machine.
