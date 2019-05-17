import numpy as np
import pandas as pd
from bokeh.models import Band, HoverTool
from tqdm import tqdm
from copy import deepcopy, copy
from bokeh.plotting import ColumnDataSource, figure
import scipy
from scipy import interp
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import resample
from ..utils import binary_metrics, dict_median


def roc_plot(fpr, tpr, tpr_ci, median=False, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", title="", errorbar=False):
    """Creates a rocplot using Bokeh.

    Parameters
    ----------
    fpr : array-like, shape = [n_samples]
        False positive rates. Calculate using roc_calculate.

    tpr : array-like, shape = [n_samples]
        True positive rates. Calculate using roc_calculate.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci]. Calculate using roc_calculate.
    """

    # Get CI
    tpr_lowci = tpr_ci[0]
    tpr_uppci = tpr_ci[1]
    tpr_medci = tpr_ci[2]
    auc = metrics.auc(fpr, tpr)

    if median == True:
        tpr = tpr_medci

    # specificity and ci-interval for HoverTool
    spec = 1 - fpr
    ci = (tpr_uppci - tpr_lowci) / 2

    # Figure
    data = {"x": fpr, "y": tpr, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec, "ci": ci}
    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

    # Figure: add line
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal distribution line")
    figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (Train)", source=source)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb = fpr[idx]
        tpr_eb = tpr[idx]
        tpr_lowci_eb = tpr_lowci[idx]
        tpr_uppci_eb = tpr_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr[i] == 1:
                    fpr_eb = 0
                    tpr_eb = 1
                    tpr_lowci_eb = 1
                    tpr_uppci_eb = 1

        roc_whisker_line = fig.multi_line([[fpr_eb, fpr_eb]], [[tpr_lowci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        roc_whisker_bot = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_lowci_eb, tpr_lowci_eb]], line_color="black")
        roc_whisker_top = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_uppci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        fig.circle([fpr_eb], [tpr_eb], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Change font size
    fig.title.text_font_size = "11pt"
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size
    fig.legend.label_text_font = "10pt"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False
    return fig


def roc_calculate(Ytrue, Yscore, bootnum=1000, metric=None, val=None, parametric=True):
    """Calculates required metrics for the roc plot function (fpr, tpr, and tpr_ci).

    Parameters
    ----------
    Ytrue : array-like, shape = [n_samples]
        Binary label for samples (0s and 1s)

    Yscore : array-like, shape = [n_samples]
        Predicted y score for samples

    Returns
    ----------------------------------
    fpr : array-like, shape = [n_samples]
        False positive rates.

    tpr : array-like, shape = [n_samples]
        True positive rates.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci].
    """

    # Get fpr, tpr
    fpr, tpr, threshold = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)

    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])

    # if metric is provided, calculate stats
    if metric is not None:
        specificity, sensitivity, threshold = get_spec_sens_cuttoff(Ytrue, Yscore, metric, val)
        stats = get_stats(Ytrue, Yscore, specificity, parametric)
        stats["val_specificity"] = specificity
        stats["val_sensitivity"] = specificity
        stats["val_cutoffscore"] = threshold

    # bootstrap using vertical averaging
    tpr_boot = []
    boot_stats = []
    for i in range(bootnum):
        # Resample and get tpr, fpr
        Ytrue_res, Yscore_res = resample(Ytrue, Yscore)
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue_res, Yscore_res, pos_label=1, drop_intermediate=False)

        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)

        # if metric is provided, calculate stats
        if metric is not None:
            stats_res = get_stats(Ytrue_res, Yscore_res, specificity, parametric)
            boot_stats.append(stats_res)

    # Get CI for bootstat
    if metric is not None:
        bootci_stats = {}
        for i in boot_stats[0].keys():
            stats_i = [k[i] for k in boot_stats]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                uppci = np.nan
            bootci_stats[i] = [lowci, uppci]

    # Get CI for tpr
    tpr_lowci = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_uppci = np.percentile(tpr_boot, 97.5, axis=0)
    tpr_medci = np.percentile(tpr_boot, 50, axis=0)

    # Add the starting 0
    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)
    tpr_lowci = np.insert(tpr_lowci, 0, 0)
    tpr_uppci = np.insert(tpr_uppci, 0, 0)
    tpr_medci = np.insert(tpr_medci, 0, 0)

    # Concatenate tpr_ci
    tpr_ci = np.array([tpr_lowci, tpr_uppci, tpr_medci])

    if metric is None:
        return fpr, tpr, tpr_ci
    else:
        return fpr, tpr, tpr_ci, stats, bootci_stats


def roc_plot_boot(fpr_ib, tpr_ib_ci, fpr_oob, tpr_oob_ci, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", title="", errorbar=False):
    """Creates a rocplot using Bokeh.

    Parameters
    ----------
    fpr : array-like, shape = [n_samples]
        False positive rates. Calculate using roc_calculate.

    tpr : array-like, shape = [n_samples]
        True positive rates. Calculate using roc_calculate.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci]. Calculate using roc_calculate.
    """

    # Get CI
    tpr_ib = tpr_ib_ci[0]
    tpr = tpr_ib
    tpr_ib_lowci = tpr_ib_ci[1]
    tpr_ib_uppci = tpr_ib_ci[2]
    fpr = np.insert(fpr_ib[0], 0, 0)
    auc_ib = metrics.auc(fpr, tpr_ib)

    # add oob median CI
    tpr_oob = tpr_oob_ci[0]
    tpr_oob_lowci = tpr_oob_ci[1]
    tpr_oob_uppci = tpr_oob_ci[2]
    auc_oob = metrics.auc(fpr, tpr_oob)

    # specificity and ci-interval for HoverTool
    spec = 1 - fpr
    ci = (tpr_oob_uppci - tpr_oob_lowci) / 2

    # Figure
    data = {"x": fpr, "y": tpr_ib, "lowci": tpr_ib_lowci, "uppci": tpr_ib_uppci, "spec": spec, "ci": ci, "x_oob": fpr, "y_oob": tpr_oob, "lowci_oob": tpr_oob_lowci, "uppci_oob": tpr_oob_uppci}
    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

    # Figure: add line
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal distribution line")
    figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (IB)", source=source)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)

    # Figure: add oob line
    figline2 = fig.line("x_oob", "y_oob", color="red", line_width=3.5, alpha=0.6, legend="ROC Curve (OOB)", source=source)
    figband2 = Band(base="x_oob", lower="lowci_oob", upper="uppci_oob", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="red", source=source)
    fig.add_layout(figband2)

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb = fpr[idx]
        tpr_eb = tpr[idx]
        tpr_lowci_eb = tpr_ib_lowci[idx]
        tpr_uppci_eb = tpr_ib_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr[i] == 1:
                    fpr_eb = 0
                    tpr_eb = 1
                    tpr_lowci_eb = 1
                    tpr_uppci_eb = 1

        roc_whisker_line = fig.multi_line([[fpr_eb, fpr_eb]], [[tpr_lowci_eb, tpr_uppci_eb]], line_alpha=1, line_width=2, line_color="darkgreen")
        roc_whisker_bot = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_lowci_eb, tpr_lowci_eb]], line_width=2, line_color="darkgreen")
        roc_whisker_top = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_uppci_eb, tpr_uppci_eb]], line_width=2, line_alpha=1, line_color="darkgreen")
        fig.circle([fpr_eb], [tpr_eb], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb2 = fpr[idx]
        tpr_eb2 = tpr_oob[idx]
        tpr_lowci_eb2 = tpr_oob_lowci[idx]
        tpr_uppci_eb2 = tpr_oob_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr_oob[i] == 1:
                    fpr_eb2 = 0
                    tpr_eb2 = 1
                    tpr_lowci_eb2 = 1
                    tpr_uppci_eb2 = 1

        roc_whisker_line2 = fig.multi_line([[fpr_eb2, fpr_eb2]], [[tpr_lowci_eb2, tpr_uppci_eb2]], line_width=2, line_alpha=1, line_color="darkred")
        roc_whisker_bot2 = fig.multi_line([[fpr_eb2 - 0.03, fpr_eb2 + 0.03]], [[tpr_lowci_eb2, tpr_lowci_eb2]], line_width=2, line_color="darkred")
        roc_whisker_top2 = fig.multi_line([[fpr_eb2 - 0.03, fpr_eb2 + 0.03]], [[tpr_uppci_eb2, tpr_uppci_eb2]], line_width=2, line_alpha=1, line_color="darkred")
        fig.circle([fpr_eb2], [tpr_eb2], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Change font size
    fig.title.text_font_size = "11pt"
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size
    fig.legend.label_text_font = "10pt"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False
    return fig


def roc_calculate_boot(model, Xtrue, Ytrue, Yscore, bootnum=1000, metric=None, val=None, parametric=True):
    """Calculates required metrics for the roc plot function (fpr, tpr, and tpr_ci).

    Parameters
    ----------
    Ytrue : array-like, shape = [n_samples]
        Binary label for samples (0s and 1s)

    Yscore : array-like, shape = [n_samples]
        Predicted y score for samples

    Returns
    ----------------------------------
    fpr : array-like, shape = [n_samples]
        False positive rates.

    tpr : array-like, shape = [n_samples]
        True positive rates.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci].
    """
    # model copy
    model_boot = copy(model)

    # Get fpr, tpr
    fpr, tpr, threshold = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)

    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])

    # if metric is provided, calculate stats
    if metric is not None:
        specificity, sensitivity, threshold = get_spec_sens_cuttoff(Ytrue, Yscore, metric, val)
        stats = get_stats(Ytrue, Yscore, specificity, parametric)
        stats["val_specificity"] = specificity
        stats["val_sensitivity"] = specificity
        stats["val_cutoffscore"] = threshold

    # bootstrap using vertical averaging to linspace
    mean_fpr = np.linspace(0, 1, 1000)
    # understand location
    x_loc = pd.DataFrame(Xtrue)
    x0_loc = list(x_loc[Ytrue == 0].index)
    x1_loc = list(x_loc[Ytrue == 1].index)
    x_loc_ib_dict = {k: [] for k in list(x_loc.index)}
    x_loc_oob_dict = {k: [] for k in list(x_loc.index)}
    # stratified resample
    x0 = Xtrue[Ytrue == 0]
    x1 = Xtrue[Ytrue == 1]
    x0_idx = list(range(len(x0)))
    x1_idx = list(range(len(x1)))
    tpr_ib = []
    fpr_ib = []
    stat_ib_boot = []
    median_ib = []
    tpr_oob = []
    fpr_oob = []
    stat_oob_boot = []
    median_oob = []
    manw_pval = []
    for i in tqdm(range(bootnum)):
        # resample
        x0_idx_ib = resample(x0_idx)
        x1_idx_ib = resample(x1_idx)
        x0_idx_oob = list(set(x0_idx) - set(x0_idx_ib))
        x1_idx_oob = list(set(x1_idx) - set(x1_idx_ib))
        # get x
        x0_ib = x0[x0_idx_ib]
        x1_ib = x1[x1_idx_ib]
        x0_oob = x0[x0_idx_oob]
        x1_oob = x1[x1_idx_oob]
        x_ib = np.concatenate((x0_ib, x1_ib))
        x_oob = np.concatenate((x0_oob, x1_oob))
        # get y
        y0_ib = np.zeros(len(x0_idx_ib))
        y1_ib = np.ones(len(x1_idx_ib))
        y0_oob = np.zeros(len(x0_idx_oob))
        y1_oob = np.ones(len(x1_idx_oob))
        y_ib = np.concatenate((y0_ib, y1_ib))
        y_oob = np.concatenate((y0_oob, y1_oob))
        # train and test model
        ypred_ib = model_boot.train(x_ib, y_ib)
        ypred_oob = model_boot.test(x_oob)
        # get median ypred per group
        ypred_ib_0 = ypred_ib[: len(x0_idx_ib)]
        ypred_ib_1 = ypred_ib[len(x0_idx_ib):]
        median_ib.append([np.median(ypred_ib_0), np.median(ypred_ib_1)])
        # get average ypred
        for i in range(len(ypred_ib_0)):
            idx_res = x0_idx_ib[i]
            idx_true = x0_loc[idx_res]
            x_loc_ib_dict[idx_true].append(ypred_ib_0[i])
        for i in range(len(ypred_ib_1)):
            idx_res = x1_idx_ib[i]
            idx_true = x1_loc[idx_res]
            x_loc_ib_dict[idx_true].append(ypred_ib_1[i])
        # get ib fpr, tpr, stats
        fpri, tpri, _ = metrics.roc_curve(y_ib, ypred_ib, pos_label=1, drop_intermediate=False)
        fpr_ib.append(mean_fpr)
        tpr_ib.append(interp(mean_fpr, fpri, tpri))
        # tpr_ib[-1][0] = 0.0
        # if metric is provided, calculate stats
        if metric is not None:
            stats_resi = get_stats(y_ib, ypred_ib, specificity, parametric)
            stat_ib_boot.append(stats_resi)
        # get median ypred per group
        ypred_oob_0 = ypred_oob[: len(x0_idx_oob)]
        ypred_oob_1 = ypred_oob[len(x0_idx_oob):]
        median_oob.append([np.median(ypred_oob_0), np.median(ypred_oob_1)])
        # get average ypred
        for i in range(len(ypred_oob_0)):
            idx_res = x0_idx_oob[i]
            idx_true = x0_loc[idx_res]
            x_loc_oob_dict[idx_true].append(ypred_oob_0[i])
        for i in range(len(ypred_oob_1)):
            idx_res = x1_idx_oob[i]
            idx_true = x1_loc[idx_res]
            x_loc_oob_dict[idx_true].append(ypred_oob_1[i])
        # get oob
        fpro, tpro, _ = metrics.roc_curve(y_oob, ypred_oob, pos_label=1, drop_intermediate=False)
        fpr_oob.append(mean_fpr)
        tpr_oob.append(interp(mean_fpr, fpro, tpro))
        tpr_oob[-1][0] = 0.0
        # if metric is provided, calculate stats
        if metric is not None:
            stats_reso = get_stats(y_oob, ypred_oob, specificity, parametric)
            stat_oob_boot.append(stats_reso)
        # manu
        manw_pval_ib = scipy.stats.mannwhitneyu(ypred_ib_0, ypred_ib_1, alternative="two-sided")[1]
        manw_pval_oob = scipy.stats.mannwhitneyu(ypred_oob_0, ypred_oob_1, alternative="two-sided")[1]
        manw_pval.append([manw_pval_ib, manw_pval_oob])

    # Get CI for bootstat ib
    if metric is not None:
        stat_ib = {}
        for i in stat_ib_boot[0].keys():
            stats_i = [k[i] for k in stat_ib_boot]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                medci = np.percentile(stats_i, 50)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                medci = np.nan
                uppci = np.nan
            stat_ib[i] = [medci, lowci, uppci]

    # Get CI for bootstat oob
    if metric is not None:
        stat_oob = {}
        for i in stat_oob_boot[0].keys():
            stats_i = [k[i] for k in stat_oob_boot]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                medci = np.percentile(stats_i, 50)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                medci = np.nan
                uppci = np.nan
            stat_oob[i] = [medci, lowci, uppci]

    # Get CI for tpr
    tpr_oob_lowci = np.percentile(tpr_oob, 2.5, axis=0)
    tpr_oob_medci = np.percentile(tpr_oob, 50, axis=0)
    tpr_oob_uppci = np.percentile(tpr_oob, 97.5, axis=0)

    # Add the starting 0
    tpr_oob_lowci = np.insert(tpr_oob_lowci, 0, 0)
    tpr_oob_medci = np.insert(tpr_oob_medci, 0, 0)
    tpr_oob_uppci = np.insert(tpr_oob_uppci, 0, 0)

    # Concatenate tpr_ci
    tpr_oob_ci = np.array([tpr_oob_medci, tpr_oob_lowci, tpr_oob_uppci])

    # Get CI for tpr
    tpr_ib_lowci = np.percentile(tpr_ib, 2.5, axis=0)
    tpr_ib_medci = np.percentile(tpr_ib, 50, axis=0)
    tpr_ib_uppci = np.percentile(tpr_ib, 97.5, axis=0)

    # Add the starting 0
    tpr_ib_lowci = np.insert(tpr_ib_lowci, 0, 0)
    tpr_ib_medci = np.insert(tpr_ib_medci, 0, 0)
    tpr_ib_uppci = np.insert(tpr_ib_uppci, 0, 0)

    # Get median score per boot
    median_y_ib = x_loc_ib_dict
    median_y_oob = x_loc_oob_dict

    # Concatenate tpr_ci
    tpr_ib_ci = np.array([tpr_ib_medci, tpr_ib_lowci, tpr_ib_uppci])

    if metric is None:
        return fpr, tpr, tpr_ci
    else:
        return fpr_ib, tpr_ib_ci, stat_ib, median_ib, fpr_oob, tpr_oob_ci, stat_oob, median_oob, stats, median_y_ib, median_y_oob, manw_pval


def get_sens_spec(Ytrue, Yscore, cuttoff_val):
    """Get sensitivity and specificity from cutoff value."""
    Yscore_round = np.where(np.array(Yscore) > cuttoff_val, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(Ytrue, Yscore_round).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def get_sens_cuttoff(Ytrue, Yscore, specificity_val):
    """Get sensitivity and cuttoff value from specificity."""
    fpr0 = 1 - specificity_val
    fpr, sensitivity, thresholds = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)
    idx = np.abs(fpr - fpr0).argmin()  # this find the closest value in fpr to fpr0
    # Check that this is not a perfect roc curve
    # If it is perfect, allow sensitivity = 1, rather than 0
    if specificity_val == 1 and sensitivity[idx] == 0:
        for i in range(len(fpr)):
            if fpr[i] == 1 and sensitivity[i] == 1:
                return 1, 0.5
    return sensitivity[idx], thresholds[idx]


def get_spec_sens_cuttoff(Ytrue, Yscore, metric, val):
    """Return specificity, sensitivity, cutoff value provided the metric and value used."""
    if metric == "specificity":
        specificity = val
        sensitivity, threshold = get_sens_cuttoff(Ytrue, Yscore, val)
    elif metric == "cutoffscore":
        threshold = val
        sensitivity, specificity = get_sens_spec(Ytrue, Yscore, val)
    return specificity, sensitivity, threshold


def get_stats(Ytrue, Yscore, specificity, parametric):
    """Calculates binary metrics given the specificity."""
    sensitivity, cutoffscore = get_sens_cuttoff(Ytrue, Yscore, specificity)
    stats = binary_metrics(Ytrue, Yscore, cut_off=cutoffscore, parametric=parametric)
    return stats
