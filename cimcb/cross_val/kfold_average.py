import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed
import multiprocessing
import timeit
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, Circle, HoverTool, TapTool, LabelSet, Rect, LinearColorMapper, MultiLine
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import ParameterGrid
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics, dict_mean


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

    def __init__(self, model, X, Y, param_dict, folds=10, n_cores=-1, mc=0):
        for key, value in param_dict.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                param_dict[key] = [value]

        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds)
        self.crossval_idx = StratifiedKFold(n_splits=folds)
        self.param_dict2 = {}
        for key, value in param_dict.items():
            if len(value) > 1:
                self.param_dict2 = {**self.param_dict2, **{key: value}}
        self.param_list2 = list(ParameterGrid(self.param_dict2))
        if isinstance(n_cores, int) is False:
            raise ValueError("n_cores needs to be an integer.")
        self.n_cores = n_cores

        self.mc = mc
        if self.mc == 0:
            self.mc = 1

    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""

        # Start Timer
        start = timeit.default_timer()

        # Set number of cores for parallel
        max_num_cores = multiprocessing.cpu_count()
        if self.n_cores > max_num_cores:
            self.n_cores = -1
            print("Number of cores set too high. It will be set to the max number of cores in the system.")
        if self.n_cores == -1:
            self.n_cores = max_num_cores
            print("Number of cores set to: {}".format(max_num_cores))

        # Loop over parameter_list
        def processInput(i):
            # Empty list
            ypred_full_i = []
            ypred_cv_i = []
            # Set hyper - parameters
            params_i = self.param_list[i]
            model_i = self.model()
            model_i.set_params(params_i)
            # Split into train and test
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
            k = model_i.k
            ypredcvfull = [ypred_full_i, ypred_cv_i, k]
            return ypredcvfull

        def mc_loop(j):
            ypred_fullcv = Parallel(n_jobs=self.n_cores)(delayed(processInput)(i) for i in range(len(self.param_list)))
            return ypred_fullcv

        self.avstat = []
        for i in tqdm(range(0, self.mc)):
            ypred = mc_loop(1)
            self.avstat.append(ypred)

        self.ypred_full = []
        self.ypred_cv = []
        self.k = []
        for i in self.avstat:
            ypred_full = []
            ypred_cv = []
            k = []
            for j in i:
                ypred_full.append(j[0])
                ypred_cv.append(j[1])
                k.append(j[2])
            self.ypred_full.append(ypred_full)
            self.ypred_cv.append(ypred_cv)
            self.k.append(k)

        stop = timeit.default_timer()
        self.parallel_time = stop - start
        print("Time: ", self.parallel_time)

    def calc_ypred_epoch(self):
        """Calculates ypred full and ypred cv for each epoch (edge case)."""

        def mc_loop(j):

            # Store Ypred
            Y_full = []
            Y_cv = []

            # Set hyper-parameters
            param = self.param_list[-1]
            model = self.model(**param)

            # Get crossval train + test
            fold_split = []
            for train, test in self.crossval_idx.split(self.X, self.Y):
                fold_split.append((train, test))

            # Split into train and test
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

            # Put ypred into standard format
            epoch_list = []
            for i in self.param_list2:
                for k, v in i.items():
                    epoch_list.append(v - 1)

            self.ypred_full = []
            self.ypred_cv = []
            for i in epoch_list:
                ypred_full_i = []
                ypred_cv_i = []
                for j in range(self.folds):
                    ypred_full_i.append([Y_full[j][0], Y_full[j][1][i]])
                    ypred_cv_i.append([Y_cv[j][0], Y_cv[j][1][i]])
                # Append ypred to full/cv
                self.ypred_full.append(ypred_full_i)
                self.ypred_cv.append(ypred_cv_i)

            # delete k later
            self.k = []
            for i in epoch_list:
                self.k.append(model.k)

            ypredcvfull = [self.ypred_full, self.ypred_cv, self.k]
            return ypredcvfull

        Y_full = []
        Y_cv = []
        k = []
        for i in tqdm(range(0, self.mc)):
            a, b, c = mc_loop(1)
            Y_full.append(a)
            Y_cv.append(b)
            k.append(c)

        self.ypred_full = Y_full
        self.ypred_cv = Y_cv
        self.k = k

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        stats_list = []
        for i in range(len(self.param_list)):
            # Get all binary metrics
            full_loop = []
            cv_loop = []
            for k in range(len(self.ypred_full)):
                for j in range(len(self.ypred_full[k][i])):
                    full = binary_metrics(self.ypred_full[k][i][j][0], self.ypred_full[k][i][j][1], parametric=self.model.parametric, k=self.k[k][i])
                    cv = binary_metrics(self.ypred_cv[k][i][j][0], self.ypred_cv[k][i][j][1], parametric=self.model.parametric, k=self.k[k][i])
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
        self.table = self._format_table(stats_list)  # Transpose, Add headers
        return self.table

    def run(self):
        """Runs all functions prior to plot."""
        # Check that param_dict is not for epochs
        # Epoch is a special case
        check_epoch = []
        for i in self.param_dict2.keys():
            check_epoch.append(i)
        if check_epoch == ["epochs"]:
            # Get epoch max
            epoch_list = []
            for i in self.param_list2:
                for k, v in i.items():
                    epoch_list.append(v)
            # Print and Calculate
            self.calc_ypred_epoch()
            print("returning stats at 'x' epoch interval during training until epoch={}.".format(epoch_list[-1]))
        else:
            self.calc_ypred()
        self.calc_stats()

    def _format_table(self, stats_list):
        """Make stats pretty (pandas table -> proper names in columns)."""
        table = pd.DataFrame(stats_list).T
        param_list_string = []
        for i in range(len(self.param_list)):
            param_list_string.append(str(self.param_list[i]))
        table.columns = param_list_string
        return table

    def plot(self, metric="r2q2", scale=1, color_scaling="linear", rotate_xlabel=True):
        """Create a full/cv plot using based on metric selected.

        Parameters
        ----------
        metric : string, (default "r2q2")
            metric has to be either "r2q2", "auc", "acc", "f1score", "prec", "sens", or "spec".
        """
        # Check model is parametric if using 'r2q2'
        if metric == "r2q2" and self.model.parametric == False:
            print("metric changed from 'r2q2' to 'auc' as the model is non-parametric.")
            metric = "auc"

        # Plot based on the number of parameters
        if len(self.param_dict2) == 1:
            fig = self._plot_param1(metric=metric, scale=scale, rotate_xlabel=rotate_xlabel)
        elif len(self.param_dict2) == 2:
            fig = self._plot_param2(metric=metric, scale=scale, color_scaling=color_scaling)
        else:
            raise ValueError("plot function only works for 1 or 2 parameters, there are {}.".format(len(self.param_dict2)))

        # Show plot
        output_notebook()
        show(fig)

    def _plot_param1(self, metric="r2q2", scale=1, rotate_xlabel=True):
        """Used for plot function if the number of parameters is 1."""
        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AIC", "AUC", "BIC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY", "SSE"])
        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full = self.table.iloc[2 * metric_idx + 1]
        cv = self.table.iloc[2 * metric_idx]
        diff = abs(full - cv)
        full_text = self.table.iloc[2 * metric_idx + 1].name
        cv_text = self.table.iloc[2 * metric_idx].name
        diff_text = "DIFFERENCE " + "(" + full_text + " - " + cv_text + ")"

        # round full, cv, and diff for hovertool
        full_hover = []
        cv_hover = []
        diff_hover = []
        for j in range(len(full)):
            full_hover.append("%.2f" % round(full[j], 2))
            cv_hover.append("%.2f" % round(cv[j], 2))
            diff_hover.append("%.2f" % round(diff[j], 2))

        # get key, values (as string) from param_dict (key -> title, values -> x axis values)
        for k, v in self.param_dict2.items():
            key_title = k
            key_xaxis = k
            values = v
        values_string = [str(i) for i in values]
        values_string = []
        for i in values:
            if i == 0:
                values_string.append(str(i))
            elif 0.0001 > i:
                values_string.append("%0.2e" % i)
            elif 10000 < i:
                values_string.append("%0.2e" % i)
            else:
                values_string.append(str(i))

        # if parameter starts with n_ e.g. n_components change title to 'no. of components', xaxis to 'components'
        if key_title.startswith("n_"):
            key_title = "no. of " + key_title[2:]
            key_xaxis = key_xaxis[2:-1]

        # store data in ColumnDataSource for Bokeh
        data = dict(full=full, cv=cv, diff=diff, full_hover=full_hover, cv_hover=cv_hover, diff_hover=diff_hover, values_string=values_string)
        source = ColumnDataSource(data=data)

        fig1_yrange = (min(diff) - max(0.1 * (min(diff)), 0.07), max(diff) + max(0.1 * (max(diff)), 0.07))
        fig1_xrange = (min(cv) - max(0.1 * (min(cv)), 0.07), max(cv) + max(0.1 * (max(cv)), 0.07))
        fig1_title = diff_text + " vs. " + cv_text

        # Plot width/height
        width = int(485 * scale)
        height = int(405 * scale)

        # Figure 1 (DIFFERENCE (R2 - Q2) vs. Q2)
        fig1 = figure(x_axis_label=cv_text, y_axis_label=diff_text, title=fig1_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=fig1_yrange, x_range=fig1_xrange, plot_width=width, plot_height=height)

        # Figure 1: Add a line
        fig1_line = fig1.line(cv, diff, line_width=2, line_color="black", line_alpha=0.25)

        # Figure 1: Add circles (interactive click)
        fig1_circ = fig1.circle("cv", "diff", size=12, alpha=0.7, color="green", source=source)
        fig1_circ.selection_glyph = Circle(fill_color="green", line_width=2, line_color="black")
        fig1_circ.nonselection_glyph.fill_color = "green"
        fig1_circ.nonselection_glyph.fill_alpha = 0.4
        fig1_circ.nonselection_glyph.line_color = "white"
        # Add values as in text labels
        # fig1_labels = LabelSet(x="cv", y="diff", text="values_string", level="glyph", source=source, render_mode="canvas", x_offset=-4, y_offset=-7, text_font_size="10pt", text_color="white")
        # fig1.add_layout(fig1_labels)

        # Figure 1: Add hovertool
        fig1.add_tools(HoverTool(renderers=[fig1_circ], tooltips=[(key_xaxis, "@values_string"), (full_text, "@full_hover"), (cv_text, "@cv_hover"), ("Diff", "@diff_hover")]))

        # Figure 1: Extra formating
        fig1.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig1.title.text_font_size = "12pt"
            fig1.xaxis.axis_label_text_font_size = "10pt"
            fig1.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig1.title.text_font_size = "10pt"
            fig1.xaxis.axis_label_text_font_size = "9pt"
            fig1.yaxis.axis_label_text_font_size = "9pt"

        # Figure 2: full/cv
        fig2_title = full_text + " & " + cv_text + " vs. " + key_title
        fig2 = figure(x_axis_label=key_xaxis, y_axis_label="Value", title=fig2_title, plot_width=width, plot_height=height, x_range=pd.unique(values_string), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        # Figure 2: add full
        fig2_line_full = fig2.line(values_string, full, line_color="red", line_width=2)
        fig2_circ_full = fig2.circle("values_string", "full", line_color="red", fill_color="white", fill_alpha=1, size=8, source=source, legend=full_text)
        fig2_circ_full.selection_glyph = Circle(line_color="red", fill_color="white", line_width=2)
        fig2_circ_full.nonselection_glyph.line_color = "red"
        fig2_circ_full.nonselection_glyph.fill_color = "white"
        fig2_circ_full.nonselection_glyph.line_alpha = 0.4

        # Figure 2: add cv
        fig2_line_cv = fig2.line(values_string, cv, line_color="blue", line_width=2)
        fig2_circ_cv = fig2.circle("values_string", "cv", line_color="blue", fill_color="white", fill_alpha=1, size=8, source=source, legend=cv_text)
        fig2_circ_cv.selection_glyph = Circle(line_color="blue", fill_color="white", line_width=2)
        fig2_circ_cv.nonselection_glyph.line_color = "blue"
        fig2_circ_cv.nonselection_glyph.fill_color = "white"
        fig2_circ_cv.nonselection_glyph.line_alpha = 0.4

        # Add hovertool and taptool
        fig2.add_tools(HoverTool(renderers=[fig2_circ_full], tooltips=[(full_text, "@full_hover")], mode="vline"))
        fig2.add_tools(HoverTool(renderers=[fig2_circ_cv], tooltips=[(cv_text, "@cv_hover")], mode="vline"))
        fig2.add_tools(TapTool(renderers=[fig2_circ_full, fig2_circ_cv]))

        # Figure 2: Extra formating
        fig2.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig2.title.text_font_size = "12pt"
            fig2.xaxis.axis_label_text_font_size = "10pt"
            fig2.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig2.title.text_font_size = "10pt"
            fig2.xaxis.axis_label_text_font_size = "9pt"
            fig2.yaxis.axis_label_text_font_size = "9pt"

        # Rotate
        if rotate_xlabel is True:
            fig2.xaxis.major_label_orientation = np.pi / 2

        # Figure 2: legend
        if metric is "r2q2":
            fig2.legend.location = "top_left"
        else:
            fig2.legend.location = "bottom_right"

        # Create a grid and output figures
        grid = np.full((1, 2), None)
        grid[0, 0] = fig1
        grid[0, 1] = fig2
        fig = gridplot(grid.tolist(), merge_tools=True)
        return fig

    def _plot_param2(self, metric="r2q2", xlabel=None, orientation=0, alternative=False, scale=1, heatmap_xaxis_rotate=90, color_scaling="linear", line=False):

        # Need to sort out param_dict to be sorted alphabetically

        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full_score = self.table.iloc[2 * metric_idx + 1]
        cv_score = self.table.iloc[2 * metric_idx]
        diff_score = abs(full_score - cv_score)
        full_title = self.table.iloc[2 * metric_idx + 1].name
        cv_title = self.table.iloc[2 * metric_idx].name
        diff_title = full_title[:-4] + "diff"

        # round full, cv, and diff for hovertool
        full_hover = []
        cv_hover = []
        diff_hover = []
        for j in range(len(full_score)):
            full_hover.append("%.2f" % round(full_score[j], 2))
            cv_hover.append("%.2f" % round(cv_score[j], 2))
            diff_hover.append("%.2f" % round(diff_score[j], 2))

        # Get key/values
        param_keys = []
        param_values = []
        for key, value in sorted(self.param_dict2.items()):
            param_keys.append(key)
            # value_to_string = list(map(str, value))
            # param_values.append(value_to_string)
            values_string = []
            for i in value:
                if i == 0:
                    values_string.append(str(i))
                elif 0.0001 > i:
                    values_string.append("%0.2e" % i)
                elif 10000 < i:
                    values_string.append("%0.2e" % i)
                else:
                    values_string.append(str(i))

            param_values.append(values_string)

        # Get key/value combinations
        comb = list(product(param_values[0], param_values[1]))
        key0_value = [val[0] for val in comb]
        key1_value = [val[1] for val in comb]
        key0_unique = param_values[0]
        key1_unique = param_values[1]
        table = self.table
        param_dict = self.param_dict2
        param_list = self.param_list2

        # Set-up for non-linear scaling for heatmap color
        if color_scaling is "log":
            scale_full_score = np.log(full_score)
            scale_cv_score = np.log(cv_score)
            scale_diff_score = np.log(diff_score)
        elif color_scaling is "square":
            scale_full_score = full_score ** 2
            scale_cv_score = cv_score ** 2
            scale_diff_score = diff_score ** 2
        elif color_scaling is "square root":
            scale_full_score = np.sqrt(full_score)
            scale_cv_score = np.sqrt(cv_score)
            scale_diff_score = np.sqrt(diff_score)
        elif color_scaling is "log+1":
            scale_full_score = np.log(full_score + 1)
            scale_cv_score = np.log(cv_score + 1)
            scale_diff_score = np.log(diff_score + 1)
        else:
            scale_full_score = full_score
            scale_cv_score = cv_score
            scale_diff_score = diff_score

        # Basic Min_Max for heatmap (changing alpha (opaque) rather than colour)... linear from 0 to 1
        scaler = preprocessing.MinMaxScaler(feature_range=(0.02, 1))
        full_alpha = scaler.fit_transform(scale_full_score[:, np.newaxis])
        cv_alpha = scaler.fit_transform(scale_cv_score[:, np.newaxis])
        diff_alpha = scaler.fit_transform(scale_diff_score[:, np.newaxis])

        # Text for heatmaps
        full_text = []
        cv_text = []
        diff_text = []
        for i in range(len(key0_value)):
            full_text.append("%.2f" % round(full_score[i], 2))
            cv_text.append("%.2f" % round(cv_score[i], 2))
            diff_text.append("%.2f" % round(diff_score[i], 2))

        # Information for line plot
        line_key0_value = []
        for i in range(len(key0_value)):
            line_key0_value.append(key0_unique)
        line_key1_value = []
        for i in range(len(key1_value)):
            line_key1_value.append(key1_unique)

        line0_full = []
        line0_cv = []
        for i in range(len(key0_value)):
            line0_full_i = []
            line0_cv_i = []
            for j in range(len(key0_value)):
                if key0_value[i] == key0_value[j]:
                    line0_full_i.append(full_score[j])
                    line0_cv_i.append(cv_score[j])
            line0_full.append(line0_full_i)
            line0_cv.append(line0_cv_i)

        line1_full = []
        line1_cv = []
        for i in range(len(key1_value)):
            line1_full_i = []
            line1_cv_i = []
            for j in range(len(key1_value)):
                if key1_value[i] == key1_value[j]:
                    line1_full_i.append(full_score[j])
                    line1_cv_i.append(cv_score[j])
            line1_full.append(line1_full_i)
            line1_cv.append(line1_cv_i)

        # Scatterplot color and size based on key0 and key1
        color_key0 = []
        for i in range(len(key0_value)):
            for j in range(len(key0_unique)):
                if key0_value[i] == key0_unique[j]:
                    color_key0.append(j / (len(key0_unique) - 1))

        scaler_size = preprocessing.MinMaxScaler(feature_range=(4, 20))
        size_prescale_key1 = []

        for i in range(len(key1_value)):
            for j in range(len(key1_unique)):
                if key1_value[i] == key1_unique[j]:
                    size_prescale_key1.append(j / (len(key1_unique) - 1))
        scatter_size_key1 = scaler_size.fit_transform(np.array(size_prescale_key1)[:, np.newaxis])
        scatter_size_key1 = scatter_size_key1 * scale

        diff_score_neg = 1 - diff_score
        # Store information in dictionary for bokeh
        data = dict(key0_value=key1_value, key1_value=key0_value, full_score=full_score, cv_score=cv_score, diff_score=diff_score, diff_score_neg=diff_score_neg, full_alpha=full_alpha, cv_alpha=cv_alpha, diff_alpha=diff_alpha, line_key0_value=line_key0_value, line_key1_value=line_key1_value, line0_full=line0_full, line0_cv=line0_cv, line1_full=line1_full, line1_cv=line1_cv, full_text=full_text, cv_text=cv_text, diff_text=diff_text)
        source = ColumnDataSource(data=data)

        # Heatmap FULL
        p1 = figure(title=full_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p1_render = p1.rect("key1_value", "key0_value", 0.9, 0.9, color="red", alpha="full_alpha", line_color=None, source=source)

        p1_render.selection_glyph = Rect(fill_color="red", fill_alpha="full_alpha", line_width=int(3 * scale), line_color="black")
        p1_render.nonselection_glyph.fill_alpha = "full_alpha"
        p1_render.nonselection_glyph.fill_color = "red"
        p1_render.nonselection_glyph.line_color = "white"

        # Heatmap CV
        p2 = figure(title=cv_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p2_render = p2.rect("key1_value", "key0_value", 0.9, 0.9, color="blue", alpha="cv_alpha", line_color=None, source=source)

        p2_render.selection_glyph = Rect(fill_color="blue", fill_alpha="cv_alpha", line_width=int(3 * scale), line_color="black")
        p2_render.nonselection_glyph.fill_alpha = "cv_alpha"
        p2_render.nonselection_glyph.fill_color = "blue"
        p2_render.nonselection_glyph.line_color = "white"

        # Heatmap Diff
        p3 = figure(title=diff_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p3_render = p3.rect("key1_value", "key0_value", 0.9, 0.9, color="green", alpha="diff_alpha", line_color=None, source=source)

        p3_render.selection_glyph = Rect(fill_color="green", fill_alpha="diff_alpha", line_width=int(3 * scale), line_color="black")
        p3_render.nonselection_glyph.fill_alpha = "diff_alpha"
        p3_render.nonselection_glyph.fill_color = "green"
        p3_render.nonselection_glyph.line_color = "white"

        # Extra for heatmaps
        p1.plot_width = int(320 * scale)
        p1.plot_height = int(257 * scale)
        p1.grid.grid_line_color = None
        p1.axis.axis_line_color = None
        p1.axis.major_tick_line_color = None
        p1.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p1.axis.major_label_standoff = 0
        p1.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p1.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p1.title.text_font_size = str(14 * scale) + "pt"
        p1.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p2.plot_width = int(320 * scale)
        p2.plot_height = int(257 * scale)
        p2.grid.grid_line_color = None
        p2.axis.axis_line_color = None
        p2.axis.major_tick_line_color = None
        p2.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p2.axis.major_label_standoff = 0
        p2.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p2.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p2.title.text_font_size = str(14 * scale) + "pt"
        p2.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p3.plot_width = int(320 * scale)
        p3.plot_height = int(257 * scale)
        p3.grid.grid_line_color = None
        p3.axis.axis_line_color = None
        p3.axis.major_tick_line_color = None
        p3.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p3.axis.major_label_standoff = 0
        p3.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p3.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p3.title.text_font_size = str(14 * scale) + "pt"
        p3.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        text = False
        # Adding text to heatmaps
        if text is True:
            # if heatmap rect is dark, use light text and vise versa
            color_mapper_diff = LinearColorMapper(palette=["#000000", "#010101", "#fdfdfd", "#fefefe", "#ffffff"], low=0, high=1)

            label1 = LabelSet(x="key0_value", y="key1_value", text="full_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "full_alpha", "transform": color_mapper_diff})

            label2 = LabelSet(x="key0_value", y="key1_value", text="cv_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "cv_alpha", "transform": color_mapper_diff})

            label3 = LabelSet(x="key0_value", y="key1_value", text="diff_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "diff_alpha", "transform": color_mapper_diff})

            p1.add_layout(label1)
            p2.add_layout(label2)
            p3.add_layout(label3)

        p1.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[("AUC_full", "@full_text")]))
        p2.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[("AUC_CV", "@cv_text")]))
        p3.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[("AUC_diff", "@diff_text")]))

        # Scatterplot
        p4 = figure(title="Scatterplot (diff vs. cv)", x_axis_label=cv_title, y_axis_label=diff_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", x_range=(min(cv_score) - 0.03, max(cv_score) + 0.03), y_range=(min(diff_score) - 0.03, max(diff_score) + 0.03))

        color_mapper_scatter = LinearColorMapper(palette="Inferno256", low=0, high=1)

        p4_render = p4.circle("cv_score", "diff_score", size=8 * scale, alpha=0.6, color="green", source=source)
        p4_render.selection_glyph = Circle(fill_color="green", line_width=int(2 * scale), line_color="black")
        p4_render.nonselection_glyph.fill_color = "green"
        p4_render.nonselection_glyph.fill_alpha = 0.4
        p4_render.nonselection_glyph.line_color = "white"
        p4.add_tools(HoverTool(renderers=[p4_render], tooltips=[("AUC_full", "@full_text"), ("AUC_CV", "@cv_text"), ("AUC_diff", "@diff_text")]))

        p4.plot_width = int(320 * scale)
        p4.plot_height = int(257 * scale)
        p4.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p4.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p4.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p4.title.text_font_size = str(14 * scale) + "pt"

        # Line plot 1
        p5 = figure(title="Lineplot ({})".format(param_keys[0]), x_axis_label=param_keys[0], y_axis_label="Value", plot_width=int(320 * scale), plot_height=int(257 * scale), x_range=pd.unique(key0_unique), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        p5_render_1 = p5.multi_line("line_key0_value", "line1_full", line_color="red", line_width=2 * scale, source=source)
        p5_render_1.selection_glyph = MultiLine(line_color="red", line_alpha=0.8, line_width=2 * scale)
        p5_render_1.nonselection_glyph.line_color = "red"
        p5_render_1.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_2 = p5.circle("key1_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source)
        p5_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p5_render_2.nonselection_glyph.line_color = "red"
        p5_render_2.nonselection_glyph.fill_color = "white"
        p5_render_2.nonselection_glyph.line_alpha = 1 / len(key1_unique)

        p5_render_3 = p5.multi_line("line_key0_value", "line1_cv", line_color="blue", line_width=2 * scale, source=source)
        p5_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p5_render_3.nonselection_glyph.line_color = "blue"
        p5_render_3.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_4 = p5.circle("key1_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source)
        p5_render_4.selection_glyph = Circle(line_color="blue", fill_color="white")
        p5_render_4.nonselection_glyph.line_color = "blue"
        p5_render_4.nonselection_glyph.fill_color = "white"
        p5_render_4.nonselection_glyph.line_alpha = 1 / len(key1_unique)

        p5.add_tools(HoverTool(renderers=[p5_render_2], tooltips=[("AUC_full", "@full_text")]))

        p5.add_tools(HoverTool(renderers=[p5_render_4], tooltips=[("AUC_CV", "@cv_text")]))

        p5.add_tools(TapTool(renderers=[p5_render_2, p5_render_4]))

        # Line plot 2
        p6 = figure(title="Lineplot ({})".format(param_keys[1]), x_axis_label=param_keys[1], y_axis_label="Value", plot_width=int(320 * scale), plot_height=int(257 * scale), x_range=pd.unique(key1_unique), tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        p6_render_1 = p6.multi_line("line_key1_value", "line0_full", line_color="red", line_width=2 * scale, source=source)
        p6_render_1.selection_glyph = MultiLine(line_color="red", line_alpha=0.8, line_width=2 * scale)
        p6_render_1.nonselection_glyph.line_color = "red"
        p6_render_1.nonselection_glyph.line_alpha = 0.05 / len(key0_unique)

        p6_render_2 = p6.circle("key0_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source)
        p6_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p6_render_2.nonselection_glyph.line_color = "red"
        p6_render_2.nonselection_glyph.fill_color = "white"
        p6_render_2.nonselection_glyph.line_alpha = 1 / len(key0_unique)

        p6_render_3 = p6.multi_line("line_key1_value", "line0_cv", line_color="blue", line_width=2 * scale, source=source)
        p6_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p6_render_3.nonselection_glyph.line_color = "blue"
        p6_render_3.nonselection_glyph.line_alpha = 0.05 / len(key0_unique)

        p6_render_4 = p6.circle("key0_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source)
        p6_render_4.selection_glyph = Circle(line_color="blue", fill_color="white")
        p6_render_4.nonselection_glyph.line_color = "blue"
        p6_render_4.nonselection_glyph.fill_color = "white"
        p6_render_4.nonselection_glyph.line_alpha = 1 / len(key0_unique)

        p6.add_tools(HoverTool(renderers=[p6_render_2], tooltips=[("AUC_full", "@full_text")]))

        p6.add_tools(HoverTool(renderers=[p6_render_4], tooltips=[("AUC_CV", "@cv_text")]))

        p6.add_tools(TapTool(renderers=[p6_render_2, p6_render_4]))

        fig = gridplot([[p1, p2, p3], [p4, p5, p6]], merge_tools=True, toolbar_location="left", toolbar_options=dict(logo=None))

        p5.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)
        p6.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p1.title.text_font_size = str(12 * scale) + "pt"
        p2.title.text_font_size = str(12 * scale) + "pt"
        p3.title.text_font_size = str(12 * scale) + "pt"
        p4.title.text_font_size = str(12 * scale) + "pt"
        p5.title.text_font_size = str(12 * scale) + "pt"
        p6.title.text_font_size = str(12 * scale) + "pt"

        return fig
