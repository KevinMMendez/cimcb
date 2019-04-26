import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import timeit
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics, dict_mean, dict_std


class kfold_average(BaseCrossVal):
    """ Exhaustitive search over param_dict calculating binary metrics.

    Parameters
    ----------
    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).

    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.

    param_dict : dict
        List of attributes to calculate and return bootstrap confidence intervals.

    folds: : a positive integer, (default 10)
        The number of folds used in the computation.

    bootnum : a positive integer, (default 100)
        The number of bootstrap samples used in the computation for the plot.

    Methods
    -------
    Run: Runs all necessary methods prior to plot.

    Plot: Creates a R2/Q2 plot.
    """

    def __init__(self, model, X, Y, param_dict, folds=5, n_mc=1, n_boot=0, n_cores=-1):
        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, n_mc=n_mc, n_boot=n_boot, n_cores=n_cores)
        self.crossval_idx = StratifiedKFold(n_splits=folds)

    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""

        # Start Timer
        start = timeit.default_timer()

        # Actual loop including Monte-Carlo reps
        self.loop_mc = self.param_list * self.n_mc
        ypred = Parallel(n_jobs=self.n_cores)(delayed(self._calc_ypred_loop)(i) for i in tqdm(range(len(self.loop_mc))))

        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        self.ypred_full = [[] for i in range(len(self.param_list))]
        self.ypred_cv = [[] for i in range(len(self.param_list))]
        self.loop_mc_numbers = list(range(len(self.param_list))) * self.n_mc
        for i in range(len(self.loop_mc)):
            j = self.loop_mc_numbers[i]  # Location to append to
            self.ypred_full[j].append(ypred[i][0])
            self.ypred_cv[j].append(ypred[i][1])

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = stop - start
        print("Time: ", self.parallel_time)

    def _calc_ypred_loop(self, i):
        """Core component of calc_ypred."""
        # Set hyper - parameters
        params_i = self.loop_mc[i]
        model_i = self.model()
        model_i.set_params(params_i)
        # Split
        ypred_full_i = []
        ypred_cv_i = []
        for train, test in self.crossval_idx.split(self.X, self.Y):
            X_train = self.X[train, :]
            Y_train = self.Y[train]
            X_test = self.X[test, :]
            Y_test = self.Y[test]
            # Full
            model_i.train(X_train, Y_train)
            ypred_full = model_i.test(X_train)
            ypred_full_i.append([Y_train, ypred_full])
            # CV (for each fold)
            ypred_cv = model_i.test(X_test)
            ypred_cv_i.append([Y_test, ypred_cv])
        return [ypred_full_i, ypred_cv_i]

    def calc_ypred_epoch(self):
        """Calculates ypred full and ypred cv for each epoch (edge case)."""

        # Set param to the max -> Actual loop including Monte-Carlo reps
        epoch_param = [self.param_list[-1]]
        self.loop_mc = epoch_param * self.n_mc
        ypred = Parallel(n_jobs=self.n_cores)(delayed(self._calc_ypred_loop_epoch)(i) for i in tqdm(range(len(self.loop_mc))))

        self.x = ypred
        # Get epoch list
        self.epoch_list = []
        for m in self.param_list2:
            for t, v in m.items():
                self.epoch_list.append(v - 1)

        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        # Note, we need to pull out the specific epochs from the model
        self.ypred_full = [[] for i in range(len(self.epoch_list))]
        self.ypred_cv = [[] for i in range(len(self.epoch_list))]
        for i in range(len(self.loop_mc)):
            for j in range(len(self.epoch_list)):
                actual_epoch = self.epoch_list[j]
                self.ypred_full[j].append([ypred[i][0][j][0]])
                self.ypred_cv[j].append([ypred[i][1][j][0]])

    def _calc_ypred_loop_epoch(self, i):
        """Core component of calc_ypred."""
        # Set hyper-parameters
        param = self.loop_mc[-1]
        model = self.model(**param)
        # CV
        fold_split = []
        for train, test in self.crossval_idx.split(self.X, self.Y):
            fold_split.append((train, test))
        # Put ypred into standard format
        epoch_list = []
        for i in self.param_list2:
            for k, v in i.items():
                epoch_list.append(v - 1)
        # Split into train and test
        Y_full = []
        Y_cv = []
        for i in range(len(fold_split)):
            train, test = fold_split[i]
            X_train = self.X[train, :]
            Y_train = self.Y[train]
            X_test = self.X[test, :]
            Y_test = self.Y[test]
            # Full
            model.train(X_train, Y_train, epoch_ypred=True, epoch_xtest=X_test)
            Y_full_split = model.epoch.Y_train
            Y_full.append([Y_train, Y_full_split])
            Y_cv_split = model.epoch.Y_test
            Y_cv.append([Y_test, Y_cv_split])
        # Get ypred from epoch_list
        ypred_full = []
        ypred_cv = []
        for i in epoch_list:
            ypred_full_i = []
            ypred_cv_i = []
            for j in range(self.folds):
                ypred_full_i.append([Y_full[j][0], Y_full[j][1][i]])
                ypred_cv_i.append([Y_cv[j][0], Y_cv[j][1][i]])
            # Append ypred to full/cv
            ypred_full.append(ypred_full_i)
            ypred_cv.append(ypred_cv_i)
        return [ypred_full, ypred_cv]

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        # Calculate for each parameter and append
        stats_list = []
        std_list = []
        for i in range(len(self.param_list)):
            # Get all binary metrics
            full_loop = []
            cv_loop = []
            for j in range(len(self.ypred_full[i])):
                for k in range(len(self.ypred_full[i][j])):
                    full = binary_metrics(self.ypred_full[i][j][k][0], self.ypred_full[i][j][k][1], parametric=self.model.parametric)
                    cv = binary_metrics(self.ypred_cv[i][j][k][0], self.ypred_cv[i][j][k][1], parametric=self.model.parametric)
                    full_loop.append(full)
                    cv_loop.append(cv)

            # Average binary metrics
            stats_full_i = dict_mean(full_loop)
            stats_cv_i = dict_mean(cv_loop)

            # Rename columns
            stats_full_i = {k + "full": v for k, v in stats_full_i.items()}
            stats_cv_i = {k + "cv": v for k, v in stats_cv_i.items()}
            stats_cv_i["R²"] = stats_full_i.pop("R²full")
            stats_cv_i["Q²"] = stats_cv_i.pop("R²cv")

            # Combine and append
            stats_combined = {**stats_full_i, **stats_cv_i}
            stats_list.append(stats_combined)

            # Keep std if n_mc > 1
            if self.n_mc > 1:
                std_full_i = dict_std(full_loop)
                std_cv_i = dict_std(cv_loop)
                std_full_i = {k + "full": v for k, v in std_full_i.items()}
                std_cv_i = {k + "cv": v for k, v in std_cv_i.items()}
                std_cv_i["R²"] = std_full_i.pop("R²full")
                std_cv_i["Q²"] = std_cv_i.pop("R²cv")
                std_combined = {**std_full_i, **std_cv_i}
                std_list.append(std_combined)

        self.table = self._format_table(stats_list)  # Transpose, Add headers
        if self.n_mc > 1:
            self.table_std = self._format_table(std_list)  # Transpose, Add headers
        return self.table
