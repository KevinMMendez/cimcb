<img src="cimcb_logo.png" alt="drawing" width="400"/>

# cimcb
cimcb package containing the necessary tools for the statistical analysis of untargeted and targeted metabolomics data.

## Installation

### Dependencies
cimcb requires:
- Python (>=3.5)
- Bokeh (>=1.0.0)
- Keras
- NumPy (>=1.12)
- SciPy
- scikit-learn
- Statsmodels
- TensorFlow
- tqdm

### User installation
The recommend way to install cimcb and dependencies is to using ``conda``:
```console
conda install -c cimcb cimcb
```
or ``pip``:
```console
pip install cimcb
```
Alternatively, to install directly from github:
```console
pip install https://github.com/KevinMMendez/cimcb/archive/master.zip
```

### Tutorial
Open with Binders:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KevinMMendez/BinderTutorial_Workflow/master?filepath=BinderTutorial_Workflow.ipynb)

### API
For futher detail on the usage refer to the docstring.

#### cimcb.model
- [PLS_SIMPLS](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/PLS_SIMPLS.py#L14-L36): Partial least-squares regression using the SIMPLS algorithm.
- [PCR](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/PCR.py#L8-L29): Principal component regression.
- [PCLR](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/PCLR.py#L8-L29): Principal component logistic regression.
- [RF](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/RF.py#L8-L9): Random forest.
- [SVM](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/SVM.py#L8-L9): Support Vector Machine.
- [RBF_NN](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/RBF_NN.py#L8-L9): Radial basis function neural network.
- [NN_LinearLinear](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/NN_LinearLinear.py#L7-L8): 2 Layer linear-linear neural network.
- [NN_LinearLogit](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/NN_LinearLogit.py#L7-L8): 2 Layer linear-logistic neural network.
- [NN_LogitLogit](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/model/NN_LogitLogit.py#L7-L8): 2 Layer logistic-logistic neural network.

#### cimcb.plot
- [boxplot](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/boxplot.py#L8-L18): Creates a boxplot using Bokeh.
- [distribution](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/distribution.py#L6-L16): Creates a distribution plot using Bokeh.
- [pca](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/pca.py#L10-L17): Creates a PCA scores and loadings plot using Bokeh.
- [permutation_test](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/permutation_test.py#L13-L27): Creates permutation test plots using Bokeh.
- [roc_plot](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/roc.py#L11-L24): Creates a rocplot using Bokeh.
- [scatter](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/scatter.py#L6-L16): Creates a scatterplot using Bokeh.
- [scatterCI](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/plot/scatterCI.py#L7-L14): Creates a scatterCI plot using Bokeh.

#### cimcb.cross_val
- [kfold](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/cross_val/kfold.py#L14-L42): Exhaustitive search over param_dict calculating binary metrics.

#### cimcb.bootstrap
- [Perc](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/bootstrap/Perc.py#L6-L35): Returns bootstrap confidence intervals using the percentile boostrap interval.
- [BC](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/bootstrap/BC.py#L8-L37): Returns bootstrap confidence intervals using the bias-corrected boostrap interval.
- [BCA](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/bootstrap/BCA.py#L8-L36): Returns bootstrap confidence intervals using the bias-corrected and accelerated boostrap interval.

#### cimcb.utils
- [binary_metrics](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/binary_metrics.py#L5-L23): Return a dict of binary stats with the following metrics: R2, auc, accuracy, precision, sensitivity, specificity, and F1 score.
- [ci95_ellipse](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/ci95_ellipse.py#L6-L28): Construct a 95% confidence ellipse using PCA.
- [knnimpute](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/knnimpute.py#L7-L22): kNN missing value imputation using Euclidean distance.
- [load_dataXL](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/load_dataXL.py#L7-L29): Loads and validates the DataFile and PeakFile from an excel file.
- [nested_getattr](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/nested_getattr.py#L4-L5): getattr for nested attributes.
- [scale](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/scale.py#L4-L42): Scales x (which can include nans) with method: 'auto', 'pareto', 'vast', or 'level'.
- [table_check](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/table_check.py#L4-L17): Error checking for DataTable and PeakTable (used in load_dataXL).
- [univariate_2class](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/univariate_2class.py#L8-L35): Creates a table of univariate statistics (2 class).
- [wmean](https://github.com/KevinMMendez/cimcb/blob/master/cimcb/utils/wmean.py#L4-L19): Returns Weighted Mean. Ignores NaNs and handles infinite weights.

### License
cimcb is licensed under the ___ license.

### Authors
- Kevin Mendez
- [David Broadhurst](https://scholar.google.ca/citations?user=M3_zZwUAAAAJ&hl=en)

### Correspondence
Professor David Broadhurst, Director of the Centre for Integrative Metabolomics & Computation Biology at Edith Cowan University.
E-mail: d.broadhurst@ecu.edu.au

### Citation
If you would cite cimcb in a scientific publication, you can use the following: ___
