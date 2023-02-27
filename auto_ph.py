import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""

Copyright 2020 - Patrick Hall (phall@h2o.ai) and the H2O.ai team

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

"""

"""

Automatic Parsimonious Hybrids (auto_ph) is a modeling approach that attempts
to build GBMs with selected monotonicity constraints deliberately from
simpler models. AutoPH is based on public ideas from the Bank of England and
the 2004 KDD Cup:

- https://bit.ly/2KmUN2J
- https://bit.ly/3bmvVEf

"""


def glm_grid(x_names, y_name, htrain, hvalid, seed_):

    """ Wrapper function for penalized GLM with alpha and lambda search.

    :param x_names: List of inputs.
    :param y_name: Name of target variable.
    :param htrain: Training H2OFrame.
    :param hvalid: Validation H2OFrame.
    :param seed_: Random seed for better reproducibility.
    :return: Best H2OGeneralizedLinearEstimator.
    """

    alpha_opts = [0.01, 0.25, 0.5, 0.99]  # always keep some L2

    # define search criteria
    # i.e., over alpha
    # lamda search handled by lambda_search param below
    hyper_parameters = {'alpha': alpha_opts}

    # initialize grid search
    grid = H2OGridSearch(
        H2OGeneralizedLinearEstimator(family="binomial",
                                      lambda_search=True,
                                      seed=seed_),
        hyper_params=hyper_parameters)

    # execute training w/ grid search
    grid.train(y=y_name,
               x=x_names,
               training_frame=htrain,
               validation_frame=hvalid,
               seed=seed_)

    # select best model from grid search
    best_model = grid.get_grid()[0]
    del grid

    return best_model


def gbm_grid(x_names, y_name, htrain, hvalid, seed_,
             monotone_constraints_=None, hyper_params_=None,
             search_criteria_=None):

    """ Wrapper that trains a random grid of H2OGradientBoostingEstimators,
        optionally with user-designated monotonicity constraints, hyper_params,
        and search criteria.

    :param x_names: List of inputs.
    :param y_name: Name of target variable.
    :param htrain: Training H2OFrame.
    :param hvalid: Validation H2OFrame.
    :param seed_: Random seed for better reproducibility.
    :param monotone_constraints_: Dictionary of monotonicity constraints (optional).
    :param hyper_params_: Dictionary of hyperparamters over which to search (optional).
    :param search_criteria_: Dictionary of criterion for grid search (optional).
    :return: Best H2OGeneralizedLinearEstimator.
    """

    # define default random grid search parameters
    if hyper_params_ is None:

        hyper_params_ = {'ntrees': list(range(1, 500, 50)),
                         'max_depth': list(range(1, 20, 2)),
                         'sample_rate': [s / float(10) for s in range(1, 11)],
                         'col_sample_rate': [s / float(10) for s in range(1, 11)]}

    # define default search strategy
    if search_criteria_ is None:

        search_criteria_ = {'strategy': 'RandomDiscrete',
                            'max_models': 20,
                            'max_runtime_secs': 600,
                            'seed': seed_}

    # initialize grid search
    grid = H2OGridSearch(H2OGradientBoostingEstimator,
                         hyper_params=hyper_params_,
                         search_criteria=search_criteria_)

    # execute training w/ grid search
    grid.train(x=x_names,
               y=y_name,
               monotone_constraints=monotone_constraints_,
               training_frame=htrain,
               validation_frame=hvalid,
               stopping_rounds=5,
               seed=seed_)

    # select best model from grid search
    best_model = grid.get_grid()[0]
    del grid

    return best_model


def gbm_forward_select_train(orig_x_names, y_name, train, valid, seed_, next_list,
                             coef_frame, new_col_name, monotone=False, monotone_constraints_=None,
                             hyper_params_=None, search_criteria_=None):

    """Trains multiple GBMs based on forward selection, optionally with user-designated
       monotonicity constraints, hyper_params, and search criteria.

    :param orig_x_names: List of inputs to include in first model and
                         from which to start forward selection process.
    :param y_name: Name of target variable.
    :param train: Pandas training frame.
    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param next_list: List of features for forward selection process.
    :param coef_frame: Pandas frame of previous model global var. imp.
                       coefficients (tightly coupled to frame schema).
    :param new_col_name: Name in coef_frame for column for this training
                         run's global var. imp. coefficients.
    :param monotone: Whether or not to create monotonic GBMs.
    :param monotone_constraints_: Dictionary of monotonicity constraints (optional).
    :param hyper_params_: Dictionary of hyperparamters over which to search (optional).
    :param search_criteria_: Dictionary of criterion for grid search (optional).
    :return: Dictionary of: list of H2O GBM models trained in forward selection, list
             containing a coef_frame for each model, list of Shapley values for each model.
    """

    # init empty parallel lists to store results
    model_list = []
    coef_list = []
    shap_list = []

    # init loop var
    selected = orig_x_names

    for j in range(0, len(next_list) + 1):

        # init or clear local dict of monotone constraints
        mc = None

        # optionally select or generate mc
        if monotone:

            if monotone_constraints_ is None:
                # create mc anew for the current model using Pearson correlation
                names = list(valid[selected + [y_name]].corr()[y_name].index)[:-1]
                signs = list([int(i) for i in np.sign(valid[selected + [y_name]].corr()[y_name].values[:-1])])
                mc = dict(zip(names, signs))
            else:
                # select mc from user designated dict: monotone_constraints_
                mc = {name_: monotone_constraints_[name_] for name_ in selected}

        # convert training and test data to h2o format
        # necessary to ensure ordering of Shapley values matches selected
        # ensure y is treated as binomial
        htrain = h2o.H2OFrame(train[selected + [y_name]])
        htrain[y_name] = htrain[y_name].asfactor()
        hvalid = h2o.H2OFrame(valid[selected + [y_name]])

        # train model and calculate Shapley values
        print('Starting grid search %i/%i ...' % (j + 1, len(next_list)+1))
        print('Input features =', selected)
        if mc is not None:
            print('Monotone constraints =', mc)
        model_list.append(gbm_grid(selected, y_name, htrain, hvalid, seed_,
                                   monotone_constraints_=mc, hyper_params_=hyper_params_,
                                   search_criteria_=search_criteria_))
        shap_values = model_list[j].predict_contributions(hvalid).as_data_frame().values[:, :-1]
        shap_list.append(shap_values)

        # update coef_frame with current model Shapley values
        # update coef_list
        col = pd.DataFrame({new_col_name: list(np.abs(shap_values).mean(axis=0))}, index=selected)
        coef_frame.update(col)
        coef_list.append(coef_frame.copy(deep=True))  # deep copy necessary

        # retrieve AUC and update progress
        auc_ = model_list[j].auc(valid=True)
        print('Completed grid search %i/%i with AUC: %.2f ...' % (j + 1, len(next_list)+1, auc_))
        print('--------------------------------------------------------------------------------')

        # add the next most y-correlated feature
        # for the next modeling iteration
        if j < len(next_list):
            selected = selected + [next_list[j]]

    print('Done.')

    return {'MODELS': model_list, 'GLOBAL_COEFS': coef_list, 'LOCAL_COEFS': shap_list}


def plot_coefs(coef_list, model_list, title_model, column_order):

    """ Plots global var. imp. importance coefficients stored in Pandas
        frames in coef_list.

    :param coef_list: List containing global var. imp. coefficients
                      for models in model list (tightly coupled to frame schemas).
    :param model_list: List of H2O GBM models trained in forward selection.
    :param title_model: Display name of model in coefficient plot.
    :param column_order: List of column names to preserve coloring
                         from previous coefficient plots.
    """
    
    for j, frame in enumerate(coef_list):
        
        auc_ = model_list[j].auc(valid=True)
        title_ = title_model + ' Model: {j}\n GBM AUC: {auc:.2f}'.format(j=str(j + 1), auc=auc_)
        fig, ax_ = plt.subplots(figsize=(10, 8))
        _ = frame[column_order].plot(kind='barh',
                                     ax=ax_,
                                     title=title_,
                                     edgecolor=['black']*len(frame.index),
                                     colormap='cool')


def cv_model_rank(valid, seed_, model_name_list, nfolds=5):

    """ Rough implementation of CV model ranking used in 2004 KDD Cup:
    https://dl.acm.org/doi/pdf/10.1145/1046456.1046470. Evaluates model
    ranks across random folds based on multiple measures.

    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param model_name_list: A list of strings in which each token is the name
                            of the Python reference to an H2O model.
    :param nfolds: Number of folds over which to evaluate model rankings.

    :return: A Pandas frame with model ranking information.
    """

    # must be metrics supported by h2o
    # assumes binary classification classification
    metric_name_list = ['mcc', 'F1', 'accuracy', 'logloss', 'auc']

    # copy original frame
    # create reproducible folds
    temp_df = valid.copy(deep=True)
    np.random.seed(seed=seed_)
    temp_df['fold'] = np.random.randint(low=0, high=nfolds, size=temp_df.shape[0])

    # initialize the returned eval_frame
    # columns for rank added later
    columns_ = ['Fold', 'Metric']
    columns_ += [model + ' Value' for model in model_name_list]
    eval_frame = pd.DataFrame(columns=columns_)

    # loop counter
    i = 0

    # loop through folds and metrics
    for fold in sorted(temp_df['fold'].unique()):
        for metric in sorted(metric_name_list):

            # necessary for adding more than one value per loop iteration
            # and appending those to eval_frame conveniently
            val_dict = {}

            # dynamically generate and run code statements
            # to calculate metrics for each fold and model
            for model in sorted(model_name_list):
                code = 'h2o.get_model("%s").model_performance(h2o.H2OFrame(temp_df[temp_df["fold"] == %d])).%s()' \
                       % (model, fold, metric)
                key_ = model + ' Value'
                val_ = eval(code)

                # some h2o metrics are returned as a list
                # this may make an assumption about binary classification?
                if isinstance(val_, list):
                    val_ = val_[0][1]
                val_dict[key_] = val_

            # create columns to store rankings
            rank_list = list(val_dict.keys())

            # add fold label and metric name into val_dict
            # with multiple model names and metric values generated above
            # append all to eval_frame
            val_dict.update({
                'Fold': fold,
                'Metric': metric})
            eval_frame = eval_frame.append(val_dict, ignore_index=True)

            # add rankings into the same row
            # conditional on direction of metric improvement
            for val_ in sorted(rank_list):
                if eval_frame.loc[i, 'Metric'] == 'logloss':
                    eval_frame.loc[i, val_.replace(' Value', ' Rank')] = eval_frame.loc[i, rank_list].rank()[val_]
                else:
                    eval_frame.loc[i, val_.replace(' Value', ' Rank')] = \
                        eval_frame.loc[i, rank_list].rank(ascending=False)[val_]

            i += 1

    del temp_df

    return eval_frame


def cv_model_rank_select(valid, seed_, train_results, model_prefix,
                         compare_model_ids, nfolds=5):

    """ Performs CV ranking for models in model_list, as compared
        to other models in model_name_list and automatically
        selects highest ranking model across the model_list.

    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param train_results: Dict created by gbm_forward_select_train
                          containing a list of models, a list of
                          global coefficients, and a list of local
                          coefficients.
    :param model_prefix: String prefix for generated model_id's.
    :param compare_model_ids: A list of H2O model_ids.
    :param nfolds: Number of folds over which to evaluate model rankings.

    :return: Best model from model_list, it's associated
             coefficients from coef_list, and the CV rank eval_frame
             for the best model.
    """

    best_idx = 0
    rank = len(compare_model_ids) + 1
    best_model_frame = None

    for i in range(0, len(train_results['MODELS'])):

        # assign model_ids correctly
        # so models can be accessed by model_id
        # in cv_model_rank
        model_id = model_prefix + str(i+1)
        train_results['MODELS'][i].model_id = model_id
        model_name_list = compare_model_ids + [model_id]

        # perform CV rank eval for
        # current model in model list vs. all compare models
        eval_frame = cv_model_rank(valid, seed_, model_name_list, nfolds=nfolds)

        # cache CV rank of current model
        title_model_col = model_name_list[-1] + ' Rank'
        new_rank = eval_frame[title_model_col].mean()

        # determine if this model outranks
        # previous best models
        if new_rank < rank:
            best_idx = i
            best_model_frame = eval_frame
            print('Evaluated model %i/%i with rank: %.2f* ...' % (i + 1, len(train_results['MODELS']),
                                                                  new_rank))
            rank = new_rank
        else:
            print('Evaluated model %i/%i with rank: %.2f ...' % (i + 1, len(train_results['MODELS']),
                                                                 new_rank))

    # select model and coefficients
    best_model = train_results['MODELS'][best_idx]
    best_shap = train_results['LOCAL_COEFS'][best_idx]
    best_coefs = train_results['GLOBAL_COEFS'][best_idx]

    print('Done.')

    # return best model, it's associated coefficients
    # and it's CV ranking frame
    return {'BEST_MODEL': best_model,
            'BEST_LOCAL_COEFS': best_shap,
            'BEST_GLOBAL_COEFS': best_coefs,
            'METRICS': best_model_frame}


def pd_ice(x_name, valid, model, resolution=20, bins=None):

    """ Creates Pandas DataFrame containing partial dependence or ICE
        for a single input variable.

    :param x_name: Variable for which to calculate partial dependence.
    :param valid: Pandas validation frame.
    :param model: H2O model (assumes binary classifier).
    :param resolution: The number of points across the domain of xs for which
                       to calculate partial dependence, default 20.
    :param bins: List of values at which to set xs, default 20 equally-spaced
                 points between column minimum and maximum.

    :return: Pandas DataFrame containing partial dependence values.

    """

    # turn off pesky Pandas copy warning
    pd.options.mode.chained_assignment = None

    # determine values at which to calculate partial dependence
    if bins is None:
        min_ = valid[x_name].min()
        max_ = valid[x_name].max()
        by = (max_ - min_) / resolution
        # modify max and by
        # to preserve resolution and actually search up to max
        bins = np.arange(min_, (max_ + by), (by + np.round((1. / resolution) * by, 3)))

        # cache original column values
    col_cache = valid.loc[:, x_name].copy(deep=True)

    # calculate partial dependence
    # by setting column of interest to constant
    # and scoring the altered data and taking the mean of the predictions
    temp_df = valid.copy(deep=True)
    temp_df.loc[:, x_name] = bins[0]
    for j, _ in enumerate(bins):
        if j + 1 < len(bins):
            valid.loc[:, x_name] = bins[j + 1]
            temp_df = temp_df.append(valid, ignore_index=True)

    # return input frame to original cached state
    valid.loc[:, x_name] = col_cache

    # model predictions
    # probably assumes binary classification
    temp_df['partial_dependence'] = model.predict(h2o.H2OFrame(temp_df))['p1'].as_data_frame()

    return pd.DataFrame(temp_df[[x_name, 'partial_dependence']].groupby([x_name]).mean()).reset_index()


def get_percentile_dict(yhat_name, valid, id_):

    """ Returns the percentiles of a column, yhat_name, as the indices based on
        another column id_.

    :param yhat_name: Name of column in valid in which to find percentiles.
    :param valid: Pandas validation frame.
    :param id_: Validation Pandas frame containing yhat and id_.

    :return: Dictionary of percentile values and index column values.

    """

    # create a copy of frame and sort it by yhat
    sort_df = valid.copy(deep=True)
    sort_df.sort_values(yhat_name, inplace=True)
    sort_df.reset_index(inplace=True)

    # find top and bottom percentiles
    percentiles_dict = {0: sort_df.loc[0, id_], 99: sort_df.loc[sort_df.shape[0] - 1, id_]}

    # find 10th-90th percentiles
    inc = sort_df.shape[0] // 10
    for i in range(1, 10):
        percentiles_dict[i * 10] = sort_df.loc[i * inc, id_]

    return percentiles_dict


def plot_pd_ice(x_name, par_dep_frame, ax=None):

    """ Plots ICE overlayed onto partial dependence for a single variable.
    Conditionally uses user-defined axes, ticks, and labels for grouped subplots.

    :param x_name: Name of variable for which to plot ICE and partial dependence.
    :param par_dep_frame: Name of Pandas frame containing ICE and partial
                          dependence values (tightly coupled to frame schema).
    :param ax: Matplotlib axis object to use.
    """

    # for standalone plotting
    if ax is None:

        # initialize figure and axis
        fig, ax = plt.subplots()

        # plot ICE
        par_dep_frame.drop('partial_dependence', axis=1).plot(x=x_name,
                                                              colormap='cool',
                                                              ax=ax)
        # overlay partial dependence, annotate plot
        par_dep_frame.plot(title='Partial Dependence with ICE: ' + x_name,
                           x=x_name,
                           y='partial_dependence',
                           color='grey',
                           linewidth=3,
                           ax=ax)

    # for grouped subplots
    else:

        # plot ICE
        par_dep_frame.drop('partial_dependence', axis=1).plot(x=x_name,
                                                              colormap='cool',
                                                              ax=ax)

        # overlay partial dependence, annotate plot
        par_dep_frame.plot(title='Partial Dependence with ICE: ' + x_name,
                           x=x_name,
                           y='partial_dependence',
                           color='red',
                           linewidth=3,
                           ax=ax)


def hist_mean_pd_ice_plot(x_name, y_name, valid, pd_ice_dict):

    """ Plots diagnostic plot of histogram with mean line overlay
        side-by-side with partial dependence and ICE.

    :param x_name: Name of variable for which to plot ICE and partial dependence.
    :param y_name: Name of target variable.
    :param valid: Pandas validation frame.
    :param pd_ice_dict: Dict of Pandas DataFrames containing partial dependence
                        and ICE values.
    """

    # initialize figure and axis
    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=False)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1.8, wspace=0.18)

    # if variable is *not* high cardinality
    # create histogram directly
    if valid[x_name].nunique() <= 20:
        mean_df = valid[[x_name, y_name]].groupby(by=x_name).mean()
        freq, bins, _ = ax.hist(valid[x_name], color='k')

    # if variable is high cardinality
    # bin, then create hist
    else:
        temp_df = pd.concat([pd.cut(valid[x_name], pd_ice_dict[x_name][x_name] - 1, duplicates='drop'),
                             valid[y_name]], axis=1)
        mean_df = temp_df.groupby(by=x_name).mean()
        del temp_df
        freq, bins, _ = ax.hist(valid[x_name], bins=pd_ice_dict[x_name][x_name] - 1, color='k')
        bins = bins[:-1]

    # annotate hist
    ax.set_xlabel(x_name)
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram with Mean ' + y_name + ' Overlay')

    # create a new twin axis
    # on which to plot a line showing mean value
    # across hist bins
    ax1 = ax.twinx()
    _ = ax1.set_ylim((0, 1))
    _ = ax1.plot(bins, mean_df.reindex(labels=bins)[y_name], color='r')
    _ = ax1.set_ylabel('Mean ' + y_name)
    _ = ax1.legend(['Mean ' + y_name], loc=1)

    # plot PD and ICE
    plot_pd_ice(x_name,
                pd_ice_dict[x_name],
                ax2)
    _ = ax2.legend(bbox_to_anchor=(1.05, 0),
                   loc=3,
                   borderaxespad=0.)


def get_confusion_matrix(valid, y_name, yhat_name, by=None, level=None, cutoff=0.5):

    """ Creates confusion matrix from pandas DataFrame of y and yhat values, can be sliced
        by a variable and level.

        :param valid: Validation DataFrame of actual (y) and predicted (yhat) values.
        :param y_name: Name of actual value column.
        :param yhat_name: Name of predicted value column.
        :param by: By variable to slice frame before creating confusion matrix, default None.
        :param level: Value of by variable to slice frame before creating confusion matrix, default None.
        :param cutoff: Cutoff threshold for confusion matrix, default 0.5.

        :return: Confusion matrix as pandas DataFrame.
    """

    # determine levels of target (y) variable
    # sort for consistency
    level_list = list(valid[y_name].unique())
    level_list.sort(reverse=True)

    # init confusion matrix
    cm_frame = pd.DataFrame(columns=['actual: ' + str(i) for i in level_list],
                            index=['predicted: ' + str(i) for i in level_list])

    # don't destroy original data
    frame_ = valid.copy(deep=True)

    # convert numeric predictions to binary decisions using cutoff
    dname = 'd_' + str(y_name)
    frame_[dname] = np.where(frame_[yhat_name] > cutoff, 1, 0)

    # slice frame
    if (by is not None) & (level is not None):
        frame_ = frame_[valid[by] == level]

    # calculate size of each confusion matrix value
    for i, lev_i in enumerate(level_list):
        for j, lev_j in enumerate(level_list):
            cm_frame.iat[j, i] = frame_[(frame_[y_name] == lev_i) & (frame_[dname] == lev_j)].shape[0]
            # i, j vs. j, i nasty little bug ... updated 8/30/19

    return cm_frame


def air(cm_dict, reference, protected):

    """ Calculates the adverse impact ratio as a quotient between protected and
        reference group acceptance rates: protected_prop/reference_prop.
        Prints intermediate values. Tightly coupled to cm_dict.

        :param cm_dict: Dict of confusion matrices containing information
                        about reference and protected groups.
        :param reference: Name of reference group in cm_dict as a string.
        :param protected: Name of protected group in cm_dict as a string.
        :return: AIR value.
    """

    # reference group summary
    reference_accepted = float(cm_dict[reference].iat[1, 0] + cm_dict[reference].iat[1, 1])  # predicted 0's
    reference_total = float(cm_dict[reference].sum().sum())
    reference_prop = reference_accepted / reference_total
    print(reference.title() + ' proportion accepted: %.3f' % reference_prop)

    # protected group summary
    protected_accepted = float(cm_dict[protected].iat[1, 0] + cm_dict[protected].iat[1, 1])  # predicted 0's
    protected_total = float(cm_dict[protected].sum().sum())
    protected_prop = protected_accepted / protected_total
    print(protected.title() + ' proportion accepted: %.3f' % protected_prop)

    # return adverse impact ratio
    return protected_prop/reference_prop


def marginal_effect(cm_dict, reference, protected):

    """ Calculates the marginal effect as a percentage difference between a reference and
        a protected group: reference_percent - protected_percent. Prints intermediate values.
        Tightly coupled to cm_dict.

        :param cm_dict: Dict of confusion matrices containing information
                        about reference and protected groups.
        :param reference: Name of reference group in cm_dict as a string.
        :param protected: Name of protected group in cm_dict as a string.
        :return: Marginal effect value.

    """

    # reference group summary
    reference_accepted = float(cm_dict[reference].iat[1, 0] + cm_dict[reference].iat[1, 1])  # predicted 0's
    reference_total = float(cm_dict[reference].sum().sum())
    reference_percent = 100 * (reference_accepted / reference_total)
    print(reference.title() + ' accepted: %.2f%%' % reference_percent)

    # protected group summary
    protected_accepted = float(cm_dict[protected].iat[1, 0] + cm_dict[protected].iat[1, 1])  # predicted 0's
    protected_total = float(cm_dict[protected].sum().sum())
    protected_percent = 100 * (protected_accepted / protected_total)
    print(protected.title() + ' accepted: %.2f%%' % protected_percent)

    # return marginal effect
    return reference_percent - protected_percent


def smd(valid, x_name, yhat_name, reference, protected):

    """ Calculates standardized mean difference between a protected and reference group:
        (mean(yhat | x_j=protected) - mean(yhat | x_j=reference))/sigma(yhat).
        Prints intermediate values.

        :param valid: Pandas dataframe containing j and predicted (yhat) values.
        :param x_name: name of demographic column containing reference and protected group labels.
        :param yhat_name: Name of predicted value column.
        :param reference: name of reference group in x_name.
        :param protected: name of protected group in x_name.

    Returns:
       Standardized mean difference as a formatted string.

    """

    # yhat mean for j=reference
    reference_yhat_mean = valid[valid[x_name] == reference][yhat_name].mean()
    print(reference.title() + ' mean yhat: %.2f' % reference_yhat_mean)

    # yhat mean for j=protected
    protected_yhat_mean = valid[valid[x_name] == protected][yhat_name].mean()
    print(protected.title() + ' mean yhat: %.2f' % protected_yhat_mean)

    # std for yhat
    sigma = valid[yhat_name].std()
    print(yhat_name.title() + ' std. dev.:  %.2f' % sigma)

    return (protected_yhat_mean - reference_yhat_mean) / sigma
