
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.preprocessing import ToTangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

import helper_function as hf


N_ELECTRODES = 8


def create_model(weights='initial_weights.hd5', n_features=800):
    """Function to create model, required for using
    KerasClassifier and wrapp a Keras model inside a
    scikitlearn form.
    We added a weight saving/loading to remove the randomness
    of the weight initialization (for better comparison).
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            34, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(17, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax'), ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop', metrics=['accuracy'])
    if weights is None:
        model.save_weights('initial_weights.hd5')
    else:
        model.load_weights(weights)
    return model


def create_model_covariance(weights='initial_weights.hd5'):
    return create_model(weights=weights, n_features=36)


def t_confint(result_intra, key, key_result):
    conf_level = 0.95
    sample = result_intra[key_result]['acc'][key][0]
    df = len(sample) - 1
    mean = result_intra[key_result]['acc_avg'][key][0]
    sem = scipy.stats.sem(sample)
    return scipy.stats.t.interval(conf_level, df, mean, sem)


def run_intrasession_on_powers(
        result_intra, data_per_exp,
        key_estimator, key_result, k_fold, powers=np.linspace(0., 2., 10)):
    for power in powers:
        metric_affine = SPDMetricAffine(N_ELECTRODES, power_affine=power)

        estimators = []
        estimators.append(('ts', ToTangentSpace(geometry=metric_affine)))
        if key_estimator == 'lr':
            tan_estimator = LogisticRegression(
                solver='lbfgs', multi_class='multinomial', max_iter=1000)
        elif key_estimator == 'dnn':
            tan_estimator = KerasClassifier(
                build_fn=create_model_covariance, epochs=100, verbose=0)
        estimators.append(('est', tan_estimator))
        pipeline = Pipeline(estimators)

        result_intra = hf.record_results_kfold(
            data_per_exp,
            result_intra,
            model=pipeline,
            kfold=k_fold,
            key_result=key_result + '_%s' % power,
            key_data='cov')

    return result_intra


def plot_intrasession_on_powers(
        result_intra, key_result, powers, ylim=(0.92, 1.)):
    plt.figure(figsize=(16, 8))

    for i, key in enumerate(['mg*s1*', 'mg*s2*', 'rr*s1*', 'rr*s2*']):
        ax = plt.subplot(1, 4, i + 1)
        x = powers
        y = [result_intra[key_result + '_%s' % power]['acc_avg'][key]
             for power in powers]
        ax.plot(x, y)
        ax.set_ylim(ylim)

        lo = []
        hi = []
        for power in powers:
            key_result_power = key_result + '_%s' % power
            confint = t_confint(result_intra, key, key_result_power)
            lo.append(confint[0])
            hi.append(confint[1])
        ax.fill_between(x, lo, hi, color='C0', alpha=.1)

        y_logrevar = result_intra['logreg_var']['acc_avg'][key]
        ax.axhline(y=y_logrevar, color='C1', label='logreg_var')
        # confint = get_confint(key, 'logreg_var')
        # ax.fill_between(x, confint[0], confint[1], color='C1', alpha=.1)

        y_ts_dnn_cov = result_intra['ts_logreg_cov']['acc_avg'][key]
        ax.axhline(y=y_ts_dnn_cov, color='C2', label='ts_logreg_cov')
        # confint = get_confint(key, 'ts_logreg_cov')
        # ax.fill_between(x, confint[0], confint[1], color='C2', alpha=.1)

        ax.axhline(
            y=result_intra['ts_dnn_cov']['acc_avg'][key],
            color='C3', label='ts_dnn_cov')

        ax.set_title(key + ': ' + key_result)
        ax.legend()


def run_intersession_on_powers(
        result_intersess, data_per_exp, key_estimator, key_result, powers):
    for power in powers:
        metric_affine = SPDMetricAffine(N_ELECTRODES, power_affine=power)

        estimators = []
        estimators.append(('ts', ToTangentSpace(geometry=metric_affine)))
        if key_estimator == 'lr':
            tan_estimator = LogisticRegression(
                solver='lbfgs', multi_class='multinomial', max_iter=1000)
        elif key_estimator == 'dnn':
            tan_estimator = KerasClassifier(
                build_fn=create_model_covariance, epochs=100, verbose=0)
        estimators.append(('est', tan_estimator))
        pipeline = Pipeline(estimators)

        result_intersess = hf.train_test_intersession(
            data_per_exp, result_intersess,
            pipeline, key_result=key_result + '_%s' % power, data_type='cov')

    return result_intersess


def plot_intersession_on_powers(
        result_intersess, key_result, powers, ylim=(0.92, 1.)):
    plt.figure(figsize=(16, 8))

    for i, key_score in enumerate(
            ['mg*s1*-mg*s2*', 'mg*s2*-mg*s1*',
             'rr*s1*-rr*s2*', 'rr*s2*-rr*s1*']):
        ax = plt.subplot(1, 4, i + 1)
        x = powers
        y = [np.mean(
                result_intersess[key_result + '_%s' % power]['acc'][key_score])
             for power in powers]
        ax.plot(x, y)
        ax.set_ylim(ylim)

        y_logrevar = np.mean(
                result_intersess['logrec_covec']['acc'][key_score])
        ax.axhline(y=y_logrevar, color='C1', label='logrec_cov')

        y_logrevar = np.mean(
                result_intersess['ts_logrec_cov']['acc'][key_score])
        ax.axhline(y=y_logrevar, color='C2', label='ts_logrec_cov')

        y_ts_dnn_cov = np.mean(
            result_intersess['dnn_covec']['acc'][key_score])
        ax.axhline(y=y_ts_dnn_cov, color='C3', label='dnn_covec')

        y_ts_dnn_cov = np.mean(
            result_intersess['ts_dnn_cov']['acc'][key_score])
        ax.axhline(y=y_ts_dnn_cov, color='C4', label='ts_dnn_cov')

        y_mdm = np.mean(
            result_intersess['eucl_mdm']['acc'][key_score])
        ax.axhline(y=y_mdm, color='C5', label='eucl_mdm')

        y_mdm = np.mean(
            result_intersess['riemann_mdm']['acc'][key_score])
        ax.axhline(y=y_mdm, color='C6', label='riem_mdm')

        ax.set_title(key_score + ': ' + key_result)
        ax.legend()


def run_intersubj_on_powers(
        result_subj, data_per_exp, key_estimator, key_result, powers):
    for power in powers:
        metric_affine = SPDMetricAffine(N_ELECTRODES, power_affine=power)

        estimators = []
        estimators.append(('ts', ToTangentSpace(geometry=metric_affine)))
        if key_estimator == 'lr':
            tan_estimator = LogisticRegression(
                solver='lbfgs', multi_class='multinomial', max_iter=1000)
        elif key_estimator == 'dnn':
            tan_estimator = KerasClassifier(
                build_fn=create_model_covariance, epochs=100, verbose=0)
        estimators.append(('est', tan_estimator))
        pipeline = Pipeline(estimators)

        result_subj = hf.train_test_intersubjects(
            data_per_exp, result_subj,
            pipeline, key_result=key_result + '_%s' % power, data_type='cov')

    return result_subj


def plot_intersubj_on_powers(
        result_subj, key_result, powers, ylim=(0.92, 1.)):
    plt.figure(figsize=(4, 8))

    ax = plt.subplot(1, 1, 1)
    x = powers
    y = [np.mean(
            result_subj[key_result + '_%s' % power]['acc'])
         for power in powers]
    ax.plot(x, y)
    ax.set_ylim(ylim)

    y_logrevar = np.mean(
            result_subj['log_covec']['acc'])
    ax.axhline(y=y_logrevar, color='C1', label='logrec_cov')

    y_logrevar = np.mean(
            result_subj['ts_log']['acc'])
    ax.axhline(y=y_logrevar, color='C2', label='ts_logrec_cov')

    y_ts_dnn_cov = np.mean(
        result_subj['dnn_covec']['acc'])
    ax.axhline(y=y_ts_dnn_cov, color='C3', label='dnn_covec')

    y_ts_dnn_cov = np.mean(
        result_subj['ts_dnn']['acc'])
    ax.axhline(y=y_ts_dnn_cov, color='C4', label='ts_dnn_cov')

    y_mdm = np.mean(
        result_subj['mdm_eucl']['acc'])
    ax.axhline(y=y_mdm, color='C5', label='eucl_mdm')

    y_mdm = np.mean(
        result_subj['mdm_riemann']['acc'])
    ax.axhline(y=y_mdm, color='C6', label='riem_mdm')

    ax.set_title(key_result)
    ax.legend()
