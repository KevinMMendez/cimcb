import numpy as np
import pandas as pd
import math
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, Circle, HoverTool, TapTool, LabelSet, Rect, LinearColorMapper, MultiLine
from tqdm import tqdm
from itertools import product
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics


class kfold_lite(BaseCrossVal):
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

    def __init__(self, model, X, Y, param_dict, folds=10, bootnum=100):
        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, bootnum=bootnum)
        self.crossval_idx = StratifiedKFold(n_splits=folds)

    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""
        self.ypred_full = []
        self.ypred_cv = []
        for params in tqdm(range(len(self.param_list)), desc="Kfold"):
            # Set hyper-parameters
            params_i = self.param_list[params]
            model_i = self.model(**params_i)
            # Full
            model_i.train(self.X, self.Y)
            ypred_full_i = model_i.test(self.X)
            self.ypred_full.append(ypred_full_i)
            # CV (for each fold)
            ypred_cv_i = self._calc_cv_ypred(model_i, self.X, self.Y)
            self.ypred_cv.append(ypred_cv_i)

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        stats_list = []
        for i in range(len(self.param_list)):
            # Create dictionaries with binary_metrics
            stats_full_i = binary_metrics(self.Y, self.ypred_full[i], parametric=self.model.parametric)
            stats_cv_i = binary_metrics(self.Y, self.ypred_cv[i], parametric=self.model.parametric)
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
        self.calc_ypred()
        self.calc_stats()
        if self.bootnum > 1:
            self.calc_ypred_boot()
            self.calc_stats_boot()

    def calc_ypred_boot(self):
        """Calculates ypred full and ypred cv for each bootstrap resample."""
        self.ytrue_boot = []
        self.ypred_full_boot = []
        self.ypred_cv_boot = []
        for i in tqdm(range(self.bootnum), desc="Kfold Boot"):
            bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            newX = self.X[bootidx_i, :]
            newY = self.Y[bootidx_i]
            ypred_full_nboot_i = []
            ypred_cv_nboot_i = []
            for params in self.param_list:
                # Set hyper-parameters
                model_i = self.model(**params)
                # Full
                model_i.train(newX, newY)
                ypred_full_i = model_i.test(newX)
                ypred_full_nboot_i.append(ypred_full_i)
                # cv
                ypred_cv_i = self._calc_cv_ypred(model_i, newX, newY)
                ypred_cv_nboot_i.append(ypred_cv_i)
            self.ytrue_boot.append(newY)
            self.ypred_full_boot.append(ypred_full_nboot_i)
            self.ypred_cv_boot.append(ypred_cv_nboot_i)

    def calc_stats_boot(self):
        """Calculates binary statistics from ypred full and ypred cv for each bootstrap resample."""
        self.full_boot_metrics = []
        self.cv_boot_metrics = []
        for i in range(len(self.param_list)):
            stats_full_i = []
            stats_cv_i = []
            for j in range(self.bootnum):
                stats_full = binary_metrics(self.ytrue_boot[j], self.ypred_full_boot[j][i])
                stats_full_i.append(stats_full)
                stats_cv = binary_metrics(self.ytrue_boot[j], self.ypred_cv_boot[j][i])
                stats_cv_i.append(stats_cv)
            self.full_boot_metrics.append(stats_full_i)
            self.cv_boot_metrics.append(stats_cv_i)

    def _calc_cv_ypred(self, model_i, X, Y):
        """Method used to calculate ypred cv."""
        ypred_cv_i = [None] * len(Y)
        for train, test in self.crossval_idx.split(self.X, self.Y):
            X_train = X[train, :]
            Y_train = Y[train]
            X_test = X[test, :]
            model_i.train(X_train, Y_train)
            ypred_cv_i_j = model_i.test(X_test)
            # Return value to y_pred_cv in the correct position # Better way to do this
            for (idx, val) in zip(test, ypred_cv_i_j):
                ypred_cv_i[idx] = val.tolist()
        return ypred_cv_i

    def _format_table(self, stats_list):
        """Make stats pretty (pandas table -> proper names in columns)."""
        table = pd.DataFrame(stats_list).T
        param_list_string = []
        for i in range(len(self.param_list)):
            param_list_string.append(str(self.param_list[i]))
        table.columns = param_list_string
        return table

    def plot(self, metric="r2q2"):
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
        if len(self.param_dict) == 1:
            fig = self._plot_param1(metric=metric)
        elif len(self.param_dict) == 2:
            fig = self._plot_param2(metric=metric)
        else:
            raise ValueError("plot function only works for 1 or 2 parametrs, there are {}.".format(len(self.param_dict)))

        # Show plot
        output_notebook()
        show(fig)

    def _plot_param1(self, metric="r2q2"):
        """Used for plot function if the number of parameters is 1."""
        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AUC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY"])
        metric_list = np.array(["acc", "auc", "f1score", "prec", "r2q2", "sens", "spec"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full = self.table.iloc[2 * metric_idx + 1]
        cv = self.table.iloc[2 * metric_idx]
        diff = full - cv
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
        for k, v in self.param_dict.items():
            key_title = k
            key_xaxis = k
            values = v
        values_string = [str(i) for i in values]

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

        # Figure 1 (DIFFERENCE (R2 - Q2) vs. Q2)
        fig1 = figure(x_axis_label=cv_text, y_axis_label=diff_text, title=fig1_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=fig1_yrange, x_range=fig1_xrange, plot_width=485, plot_height=405)

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
        fig2 = figure(x_axis_label=key_xaxis, y_axis_label="Value", title=fig2_title, plot_width=485, plot_height=405, x_range=pd.unique(values_string), y_range=(0, 1.1), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        # Figure 2: add confidence intervals if bootnum > 1
        if self.bootnum > 1:
            lower_ci_full = []
            upper_ci_full = []
            lower_ci_cv = []
            upper_ci_cv = []
            # Get all upper, lower 95% CI (full/cv) for each specific n_component and append
            for m in range(len(self.full_boot_metrics)):
                full_boot = []
                cv_boot = []
                for k in range(len(self.full_boot_metrics[0])):
                    full_boot.append(self.full_boot_metrics[m][k][metric_title[metric_idx]])
                    cv_boot.append(self.cv_boot_metrics[m][k][metric_title[metric_idx]])
                # Calculated percentile 95% CI and append
                full_bias = np.mean(full_boot) - full[m]
                cv_bias = np.mean(cv_boot) - cv[m]
                lower_ci_full.append(np.percentile(full_boot, 2.5) - full_bias)
                upper_ci_full.append(np.percentile(full_boot, 97.5) - full_bias)
                lower_ci_cv.append(np.percentile(cv_boot, 2.5) - cv_bias)
                upper_ci_cv.append(np.percentile(cv_boot, 97.5) - cv_bias)

            # Plot as a patch
            x_patch = np.hstack((values_string, values_string[::-1]))
            y_patch_r2 = np.hstack((lower_ci_full, upper_ci_full[::-1]))
            fig2.patch(x_patch, y_patch_r2, alpha=0.10, color="red")
            y_patch_q2 = np.hstack((lower_ci_cv, upper_ci_cv[::-1]))
            fig2.patch(x_patch, y_patch_q2, alpha=0.10, color="blue")

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

        metric_list = np.array(["acc", "auc", "f1score", "prec", "r2q2", "sens", "spec"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full_score = self.table.iloc[2 * metric_idx + 1]
        cv_score = self.table.iloc[2 * metric_idx]
        diff_score = full_score - cv_score
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
        for key, value in self.param_dict.items():
            param_keys.append(key)
            value_to_string = list(map(str, value))
            param_values.append(value_to_string)

        # Get key/value combinations
        comb = list(product(param_values[0], param_values[1]))
        key0_value = [val[0] for val in comb]
        key1_value = [val[1] for val in comb]
        key0_unique = param_values[0]
        key0_unique = param_values[1]

        table = self.table
        param_dict = self.param_dict
        param_list = self.param_list
        text = False

        param_keys = []
        for key, value in param_dict.items():
            param_keys.append(key)

        param_values = []
        for i in range(len(param_list)):
            for key, value in param_list[i].items():
                param_values.append(str(value))

        # param_key[0] corresponds to key0_value
        # param_key[1] corresponds to key1_value
        key0_value = param_values[0::2]
        key1_value = param_values[1::2]

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
        # print(scale_full_score)
        # print(full_alpha)
        # Text for heatmaps
        full_text = []
        cv_text = []
        diff_text = []
        for i in range(len(key0_value)):
            full_text.append("%.2f" % round(full_score[i], 2))
            cv_text.append("%.2f" % round(cv_score[i], 2))
            diff_text.append("%.2f" % round(diff_score[i], 2))

        # Information for line plot ... requires a line plot for i in param_list
        key0_unique = param_dict[param_keys[0]]
        key1_unique = param_dict[param_keys[1]]
        key0_unique = [str(i) for i in key0_unique]
        key1_unique = [str(i) for i in key1_unique]

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
        data = dict(key0_value=key0_value, key1_value=key1_value, full_score=full_score, cv_score=cv_score, diff_score=diff_score, diff_score_neg=diff_score_neg, full_alpha=full_alpha, cv_alpha=cv_alpha, diff_alpha=diff_alpha, line_key0_value=line_key0_value, line_key1_value=line_key1_value, line0_full=line0_full, line0_cv=line0_cv, line1_full=line1_full, line1_cv=line1_cv, full_text=full_text, cv_text=cv_text, diff_text=diff_text, color_key0=color_key0, scatter_size_key1=scatter_size_key1)

        source = ColumnDataSource(data=data)

        # Heatmap FULL
        p1 = figure(title=full_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p1_render = p1.rect("key0_value", "key1_value", 0.9, 0.9, color="red", alpha="full_alpha", line_color=None, source=source)

        p1_render.selection_glyph = Rect(fill_color="red", fill_alpha="full_alpha", line_width=int(3 * scale), line_color="black")
        p1_render.nonselection_glyph.fill_alpha = "full_alpha"
        p1_render.nonselection_glyph.fill_color = "red"
        p1_render.nonselection_glyph.line_color = "white"

        # Heatmap CV
        p2 = figure(title=cv_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p2_render = p2.rect("key0_value", "key1_value", 0.9, 0.9, color="blue", alpha="cv_alpha", line_color=None, source=source)

        p2_render.selection_glyph = Rect(fill_color="blue", fill_alpha="cv_alpha", line_width=int(3 * scale), line_color="black")
        p2_render.nonselection_glyph.fill_alpha = "cv_alpha"
        p2_render.nonselection_glyph.fill_color = "blue"
        p2_render.nonselection_glyph.line_color = "white"

        # Heatmap Diff
        p3 = figure(title=diff_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys[0], y_axis_label=param_keys[1])

        p3_render = p3.rect("key0_value", "key1_value", 0.9, 0.9, color="green", alpha="diff_alpha", line_color=None, source=source)

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
        p4 = figure(title="Scatterplot (cv - diff)", x_axis_label=cv_title, y_axis_label=full_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", x_range=(min(cv_score) - 0.03, max(cv_score) + 0.03), y_range=(min(diff_score) - 0.03, max(diff_score) + 0.03))

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

        # print(line_key0_value)
        # print(line1_full)
        p5_render_1 = p5.multi_line("line_key0_value", "line1_full", line_color="red", line_width=2 * scale, source=source)
        p5_render_1.selection_glyph = MultiLine(line_color="red", line_alpha=0.8, line_width=2 * scale)
        p5_render_1.nonselection_glyph.line_color = "red"
        p5_render_1.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_2 = p5.circle("key0_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source)
        p5_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p5_render_2.nonselection_glyph.line_color = "red"
        p5_render_2.nonselection_glyph.fill_color = "white"
        p5_render_2.nonselection_glyph.line_alpha = 1 / len(key1_unique)

        p5_render_3 = p5.multi_line("line_key0_value", "line1_cv", line_color="blue", line_width=2 * scale, source=source)
        p5_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p5_render_3.nonselection_glyph.line_color = "blue"
        p5_render_3.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_4 = p5.circle("key0_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source)
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

        p6_render_2 = p6.circle("key1_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source)
        p6_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p6_render_2.nonselection_glyph.line_color = "red"
        p6_render_2.nonselection_glyph.fill_color = "white"
        p6_render_2.nonselection_glyph.line_alpha = 1 / len(key0_unique)

        p6_render_3 = p6.multi_line("line_key1_value", "line0_cv", line_color="blue", line_width=2 * scale, source=source)
        p6_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p6_render_3.nonselection_glyph.line_color = "blue"
        p6_render_3.nonselection_glyph.line_alpha = 0.05 / len(key0_unique)

        p6_render_4 = p6.circle("key1_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source)
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
