import numpy as np
import matplotlib.pyplot as plt
from aux_cleaning_functions import *
from aux_exploratory_data_analysis import detect_categorical_features
from implementations import *
from aux_preprocessing import *


def grid_search_regression(
    y_train,
    X_train,
    y_val,
    X_val,
    thershold_nans=[0.4, 0.3, 0.2],
    methods=["undersample", "oversample"],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gamma=0.05,
):
    """
    Grid search for regression
    Returns: best_params, best_f1, results_list
    """
    best_f1 = -1
    best_params = None
    results = []

    total = len(methods) * len(thershold_nans) * len(thresholds)
    run = 0
    for thr_nans in thershold_nans:
        # 1. Reset base data from original inputs
        X_tr_base = X_train.copy()
        y_tr_base = y_train.copy()
        X_val_base = X_val.copy()

        # 2. Filter NaNs
        keep_mask = remove_nan_feat(X_tr_base, threshold=thr_nans)
        X_clean_nan_tr = X_tr_base[:, keep_mask]
        X_clean_nan_val = X_val_base[:, keep_mask]

        # 3. Detect types and impute
        cat_mask, num_mask = detect_categorical_features(X_clean_nan_tr, max_unique=30)
        X_imp_tr, medians_tr = impute_numerical(
            X_clean_nan_tr, num_mask, medians=None, means=None, strategy="medians"
        )
        X_imp_tr, modes_tr = impute_categorical(X_imp_tr, cat_mask, modes=None)

        X_imp_val, _ = impute_numerical(
            X_clean_nan_val, num_mask, medians=medians_tr, strategy="medians"
        )
        X_imp_val, _ = impute_categorical(X_imp_val, cat_mask, modes=modes_tr)

        # 4. Standardize numerical features BEFORE One-Hot Encoding
        X_std_tr, means_tr, stds_tr = standardize_features(
            X_imp_tr, num_mask, means=None, stds=None
        )
        X_std_val, _, _ = standardize_features(
            X_imp_val, num_mask, means=means_tr, stds=stds_tr
        )

        # 5. One-Hot Encode
        X_ohe_tr, categories = one_hot_encode(X_std_tr, cat_mask)
        X_ohe_val, _ = one_hot_encode(X_std_val, cat_mask, categories=categories)

        # 6. Feature Selection
        corr_mask = feature_target_correlation(
            X_ohe_tr, y_tr_base, threshold_elimination=0.01
        )
        X_corr_tr = X_ohe_tr[:, corr_mask]

        colinear_mask = filter_colinear_features(X_corr_tr, y_tr_base, 0.9)
        X_proc_tr = X_corr_tr[:, colinear_mask]
        X_proc_val = X_ohe_val[:, corr_mask][:, colinear_mask]
        for meth in methods:
            run += 1
            print(
                f"\n=== Run {run}/{total//len(thresholds)} (method={meth},thr_nans = {thr_nans}) ==="
            )
            try:
                X_tr_run, y_tr_run = balance_data(
                    X_proc_tr.copy(), y_tr_base.copy(), method=meth, random_state=None
                )
                tx_tr_run = np.c_[np.ones(X_tr_run.shape[0]), X_tr_run]
                w, loss = least_squares(y_tr_run, tx_tr_run)

                # Skip invalid runs
                if np.isnan(loss) or np.isinf(loss) or loss > 10:
                    print(f"Invalid loss ({loss:.4f}), skipping.")
                    continue

                # Evaluate all thresholds for this model
                for th in thresholds:
                    preds, _ = predict_linear(X_proc_val, w, threshold=th)
                    f1 = f1_score(y_val, preds)
                    results.append((thr_nans, meth, th, f1))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (thr_nans, meth, th)
                        print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

            except Exception as e:
                print(f"Error for method={meth},thr_nans={thr_nans}: {e}")
                continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[3], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(
            f"Best F1 = {best_f1:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, threshold={best_params[2]}"
        )
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results


def grid_search_logistic_regression_base(
    y_train,
    X_train,
    y_val,
    X_val,
    thershold_nans=[0.4, 0.3, 0.2],
    methods=["undersample", "oversample"],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gammas=[0.05, 0.1, 0.5, 1.0, 5.0],
):
    """
    Grid search for logistic regression Basic
    Returns: best_params, best_f1, results_list
    """
    best_f1 = -1
    best_params = None
    results = []

    total = len(methods) * len(thershold_nans) * len(thresholds)
    run = 0
    for thr_nans in thershold_nans:
        # 1. Reset base data from original inputs
        X_tr_base = X_train.copy()
        y_tr_base = y_train.copy()
        X_val_base = X_val.copy()

        # 2. Filter NaNs
        keep_mask = remove_nan_feat(X_tr_base, threshold=thr_nans)
        X_clean_nan_tr = X_tr_base[:, keep_mask]
        X_clean_nan_val = X_val_base[:, keep_mask]

        # 3. Detect types and impute
        cat_mask, num_mask = detect_categorical_features(X_clean_nan_tr, max_unique=30)
        X_imp_tr, medians_tr = impute_numerical(
            X_clean_nan_tr, num_mask, medians=None, means=None, strategy="medians"
        )
        X_imp_tr, modes_tr = impute_categorical(X_imp_tr, cat_mask, modes=None)

        X_imp_val, _ = impute_numerical(
            X_clean_nan_val, num_mask, medians=medians_tr, strategy="medians"
        )
        X_imp_val, _ = impute_categorical(X_imp_val, cat_mask, modes=modes_tr)

        # 4. Standardize numerical features BEFORE One-Hot Encoding
        X_std_tr, means_tr, stds_tr = standardize_features(
            X_imp_tr, num_mask, means=None, stds=None
        )
        X_std_val, _, _ = standardize_features(
            X_imp_val, num_mask, means=means_tr, stds=stds_tr
        )

        # 5. One-Hot Encode
        X_ohe_tr, categories = one_hot_encode(X_std_tr, cat_mask)
        X_ohe_val, _ = one_hot_encode(X_std_val, cat_mask, categories=categories)

        # 6. Feature Selection
        corr_mask = feature_target_correlation(
            X_ohe_tr, y_tr_base, threshold_elimination=0.01
        )
        X_corr_tr = X_ohe_tr[:, corr_mask]

        colinear_mask = filter_colinear_features(X_corr_tr, y_tr_base, 0.9)
        X_proc_tr = X_corr_tr[:, colinear_mask]
        X_proc_val = X_ohe_val[:, corr_mask][:, colinear_mask]
        for meth in methods:
            for gamma in gammas:
                run += 1
                print(
                    f"\n=== Run {run}/{total//len(thresholds)} (method={meth},thr_nans = {thr_nans}, gamma = {gamma}) ==="
                )
                try:
                    X_tr_run, y_tr_run = balance_data(
                        X_proc_tr.copy(),
                        y_tr_base.copy(),
                        method=meth,
                        random_state=None,
                    )
                    tx_tr_run = np.c_[np.ones(X_tr_run.shape[0]), X_tr_run]
                    initial_w = np.zeros(tx_tr_run.shape[1])
                    w, loss = logistic_regression(
                        y_tr_run, tx_tr_run, initial_w, max_iter, gamma, tol=1e-8
                    )

                    # Skip invalid runs
                    if np.isnan(loss) or np.isinf(loss) or loss > 10:
                        print(f"Invalid loss ({loss:.4f}), skipping.")
                        continue

                    # Evaluate all thresholds for this model
                    for th in thresholds:
                        preds, _ = predict_with_threshold(X_proc_val, w, threshold=th)
                        f1 = f1_score(y_val, preds)
                        results.append((thr_nans, meth, gamma, th, f1))

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = (thr_nans, meth, gamma, th)
                            print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

                except Exception as e:
                    print(
                        f"Error for method={meth}, gamma = {gamma}, thr_nans={thr_nans}: {e}"
                    )
                    continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[-1], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(
            f"Best F1 = {best_f1:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, gamma = {best_params[2]}, threshold={best_params[3]}"
        )
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results



def grid_search_logistic_wheight(
    y_train,
    X_train,
    y_val,
    X_val,
    thershold_nans=[0.4, 0.3, 0.2],
    pos_weights=[1, 3, 5, 9],
    methods=["undersample", "oversample", "weighted"],
    lambdas=[1e-5, 1e-3, 1e-2, 1e-1],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gammas=[0.05, 0.1, 0.5, 1, 5],
    random_seed=42,
):
    """
    Grid search for weighted penalized logistic regression.
    Returns: best_params, best_f1, results_list
    """
    best_fbeta = -1
    best_params = None
    results = []
    num_base_methods = len([m for m in methods if m != "weighted"])
    num_weight_methods = 1 if "weighted" in methods else 0
    base_runs = len(thershold_nans) * len(gammas) * num_base_methods
    weighted_runs = (
        len(thershold_nans)
        * len(gammas)
        * num_weight_methods
        * len(pos_weights)
        * len(lambdas)
    )
    total_runs = base_runs + weighted_runs

    total = (
        len(methods)
        * len(thershold_nans)
        * len(pos_weights)
        * len(lambdas)
        * len(thresholds)
        * len(gammas)
    )
    run = 0
    for thr_nans in thershold_nans:

        # Reset of the datset
        X_tr_base = X_train.copy()
        y_tr_base = y_train.copy()
        X_val_base = X_val.copy()
        y_val_base = y_val.copy()

        # Remove the nans
        keep_mask = remove_nan_feat(X_tr_base, threshold=thr_nans)
        X_clean_nan_tr = X_tr_base[:, keep_mask]
        X_clean_nan_val = X_val_base[:, keep_mask]

        # Impute mean value to the nans remaining.
        cat_mask, num_mask = detect_categorical_features(X_clean_nan_tr, max_unique=30)
        X_imp_tr, medians_tr = impute_numerical(
            X_clean_nan_tr, num_mask, medians=None, means=None, strategy="medians"
        )
        X_imp_tr, modes_tr = impute_categorical(X_imp_tr, cat_mask, modes=None)

        X_imp_val, _ = impute_numerical(
            X_clean_nan_val, num_mask, medians=medians_tr, strategy="medians"
        )
        X_imp_val, _ = impute_categorical(X_imp_val, cat_mask, modes=modes_tr)

        # Standarize the dataset
        X_std_tr, means_tr, stds_tr = standardize_features(
            X_imp_tr, num_mask, means=None, stds=None
        )
        X_std_val, _, _ = standardize_features(
            X_imp_val, num_mask, means=means_tr, stds=stds_tr
        )

        # Apply one-hot codification to the encoded categorical variables.
        X_ohe_tr, categories = one_hot_encode(X_std_tr, cat_mask)
        X_ohe_val, _ = one_hot_encode(X_std_val, cat_mask, categories=categories)

        # Eliminate features with a low correlation with the predictied variable or containing a high correlation with other variables.
        mask = feature_target_correlation(
            X_ohe_tr, y_tr_base, threshold_elimination=0.01
        )
        X_cleaned_tr = X_ohe_tr[:, mask]
        mask_correlated_variables = filter_colinear_features(
            X_cleaned_tr, y_tr_base, 0.9
        )
        X_thr_train = X_cleaned_tr[:, mask_correlated_variables]
        X_thr_val = X_ohe_val[:, mask][:, mask_correlated_variables]

        # To do the grid_search faster we will work with a batch of the whole dataset

        pos_indices = np.where(y_tr_base == 1)[0]
        neg_indices = np.where(y_tr_base == 0)[0]

        if random_seed is not None:
            np.random.seed(random_seed)

        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        pos_size = int(0.3 * len(pos_indices))
        neg_size = int(0.3 * len(neg_indices))

        sub_indices = np.concatenate([pos_indices[:pos_size], neg_indices[:neg_size]])
        np.random.shuffle(sub_indices)

        X_thr_train_sub = X_thr_train[sub_indices]
        Y_tr_base_sub = y_tr_base[sub_indices]

        if np.isnan(X_thr_train).any():
            print("Warning: X_train still contains NaNs! Impute them before training.")

        for gamma in gammas:
            for meth in methods:
                if meth != "weighted":
                    run += 1
                    print(
                        f"\n=== Run {run}/{total_runs} (method={meth},thr_nans = {thr_nans},gamma = {gamma}) ==="
                    )
                    try:
                        X_meth_train, y_meth_train = balance_data(
                            X_thr_train_sub.copy(),
                            Y_tr_base_sub.copy(),
                            method=meth,
                            random_state=random_seed,
                        )
                        tx_meth_train = np.c_[
                            np.ones(X_meth_train.shape[0]), X_meth_train
                        ]
                        w, loss = logistic_regression(
                            y_meth_train,
                            tx_meth_train,
                            np.zeros(tx_meth_train.shape[1]),
                            max_iter,
                            gamma,
                            1e-5,
                        )

                        if np.isnan(loss) or np.isinf(loss) or loss > 10:
                            print(f"Invalid loss ({loss:.4f}), skipping.")
                            continue
                        for th in thresholds:
                            preds, _ = predict_with_threshold(
                                X_thr_val, w, threshold=th
                            )
                            fbeta = f_beta_score(y_val_base, preds)
                            results.append(
                                (thr_nans, meth, gamma, None, None, th, fbeta)
                            )

                            if fbeta > best_fbeta:
                                best_fbeta = fbeta
                                best_params = (thr_nans, meth, gamma, None, None, th)
                                print(
                                    f"New best F_beta = {best_fbeta:.4f} (threshold = {th:.2f})"
                                )

                    except Exception as e:
                        print(
                            f"Error for method={meth},thr_nans={thr_nans}, gamma = {gamma}: {e}"
                        )
                        continue
                else:
                    for lam in lambdas:
                        for pw in pos_weights:
                            run += 1
                            print(
                                f"\n=== Run {run}/{total_runs} (method={meth},thr_nans = {thr_nans},gamma = {gamma}, lambda = {lam}, pw = {pw}) ==="
                            )
                            # Train model
                            try:
                                loss, w = logistic_regression_weighted_gd(
                                    Y_tr_base_sub.copy(),
                                    X_thr_train_sub.copy(),
                                    lambda_=lam,
                                    gamma=gamma,
                                    pos_weight=pw,
                                    neg_weight=1.0,
                                    max_iter=max_iter,
                                    verbose=False,
                                )

                                # Skip invalid runs
                                if np.isnan(loss) or np.isinf(loss) or loss > 10:
                                    print(f"Invalid loss ({loss:.4f}), skipping.")
                                    continue

                                # Evaluate all thresholds for this model
                                for th in thresholds:
                                    preds, _ = predict_with_threshold(
                                        X_thr_val, w, threshold=th
                                    )
                                    fbeta = f_beta_score(y_val_base, preds)
                                    results.append(
                                        (thr_nans, meth, gamma, pw, lam, th, fbeta)
                                    )

                                    if fbeta > best_fbeta:
                                        best_fbeta = fbeta
                                        best_params = (
                                            thr_nans,
                                            meth,
                                            gamma,
                                            pw,
                                            lam,
                                            th,
                                        )
                                        print(
                                            f"New best F_beta = {best_fbeta:.4f} (threshold = {th:.2f})"
                                        )

                            except Exception as e:
                                print(
                                    f"Error for method={meth},thr_nans={thr_nans}, gamma = {gamma}, pos_weight={pw}, lambda_={lam}: {e}"
                                )
                                continue

    # Sort results by F_beta descending
    results.sort(key=lambda t: t[-1], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(
            f"Best F_beta = {best_fbeta:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, gamma = {best_params[2]}, pos_weight={best_params[3]}, λ={best_params[4]}, threshold={best_params[5]}"
        )
    else:
        print("No valid runs completed.")

    return best_params, best_fbeta, results


def f_beta_score(y_true, y_pred, beta=2.0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta_squared = beta**2
    f_beta = (
        (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    )
    return f_beta
