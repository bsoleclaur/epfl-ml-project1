import numpy as np
import matplotlib.pyplot as plt
from aux_cleaning_functions import *
from aux_exploratory_data_analysis import detect_categorical_features
from implementations import *
from aux_preprocessing import *

def grid_search_regression(
    y_train, X_train,
    y_val, X_val,
    thershold_nans = [0.4, 0.3, 0.2],
    methods = ["undersample", "oversample"],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gamma=0.05
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
        X_imp_tr, medians_tr = impute_numerical(X_clean_nan_tr, num_mask, medians=None, means=None, strategy='medians')
        X_imp_tr, modes_tr = impute_categorical(X_imp_tr, cat_mask, modes=None)

        X_imp_val, _ = impute_numerical(X_clean_nan_val, num_mask, medians=medians_tr, strategy='medians')
        X_imp_val, _ = impute_categorical(X_imp_val, cat_mask, modes=modes_tr)

        # 4. Standardize numerical features BEFORE One-Hot Encoding
        X_std_tr, means_tr, stds_tr = standardize_features(X_imp_tr, num_mask, means=None, stds=None)
        X_std_val, _, _ = standardize_features(X_imp_val, num_mask, means=means_tr, stds=stds_tr)

        # 5. One-Hot Encode
        X_ohe_tr, categories = one_hot_encode(X_std_tr, cat_mask)   
        X_ohe_val, _ = one_hot_encode(X_std_val, cat_mask, categories=categories)

        # 6. Feature Selection
        corr_mask = feature_target_correlation(X_ohe_tr, y_tr_base, threshold_elimination=0.01)
        X_corr_tr = X_ohe_tr[:, corr_mask]

        colinear_mask = filter_colinear_features(X_corr_tr, y_tr_base, 0.9)
        X_proc_tr = X_corr_tr[:, colinear_mask]
        X_proc_val = X_ohe_val[:, corr_mask][:, colinear_mask]
        for meth in methods: 
            run += 1
            print(f"\n=== Run {run}/{total//len(thresholds)} (method={meth},thr_nans = {thr_nans}) ===")
            try:
                X_tr_run, y_tr_run = balance_data(X_proc_tr.copy(), y_tr_base.copy(), method=meth, random_state=None)                
                w, loss = least_squares(y_tr_run, X_tr_run)
                
                # Skip invalid runs
                if np.isnan(loss) or np.isinf(loss) or loss > 10:
                    print(f"Invalid loss ({loss:.4f}), skipping.")
                    continue

                # Evaluate all thresholds for this model
                for th in thresholds:
                    preds, _ = predict_with_threshold(X_proc_val, w, threshold=th)
                    f1 = f1_score(y_val, preds)
                    results.append((thr_nans, meth, th, f1))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (thr_nans,meth, th)
                        print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

            except Exception as e:
                print(f"Error for method={meth},thr_nans={thr_nans}: {e}")
                continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[3], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(f"Best F1 = {best_f1:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, threshold={best_params[2]}")
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results

def grid_search_logistic_regression_base(
    y_train, X_train,
    y_val, X_val,
    thershold_nans = [0.4, 0.3, 0.2],
    methods = ["undersample", "oversample"],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gamma=0.05
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
        X_imp_tr, medians_tr = impute_numerical(X_clean_nan_tr, num_mask, medians=None, means=None, strategy='medians')
        X_imp_tr, modes_tr = impute_categorical(X_imp_tr, cat_mask, modes=None)

        X_imp_val, _ = impute_numerical(X_clean_nan_val, num_mask, medians=medians_tr, strategy='medians')
        X_imp_val, _ = impute_categorical(X_imp_val, cat_mask, modes=modes_tr)

        # 4. Standardize numerical features BEFORE One-Hot Encoding
        X_std_tr, means_tr, stds_tr = standardize_features(X_imp_tr, num_mask, means=None, stds=None)
        X_std_val, _, _ = standardize_features(X_imp_val, num_mask, means=means_tr, stds=stds_tr)

        # 5. One-Hot Encode
        X_ohe_tr, categories = one_hot_encode(X_std_tr, cat_mask)   
        X_ohe_val, _ = one_hot_encode(X_std_val, cat_mask, categories=categories)

        # 6. Feature Selection
        corr_mask = feature_target_correlation(X_ohe_tr, y_tr_base, threshold_elimination=0.01)
        X_corr_tr = X_ohe_tr[:, corr_mask]

        colinear_mask = filter_colinear_features(X_corr_tr, y_tr_base, 0.9)
        X_proc_tr = X_corr_tr[:, colinear_mask]
        X_proc_val = X_ohe_val[:, corr_mask][:, colinear_mask]
        for meth in methods: 
            run += 1
            print(f"\n=== Run {run}/{total//len(thresholds)} (method={meth},thr_nans = {thr_nans}) ===")
            try:
                X_tr_run, y_tr_run = balance_data(X_proc_tr.copy(), y_tr_base.copy(), method=meth, random_state=None)                
                initial_w =  np.zeros(X_tr_run.shape[1]+1)
                w, loss = logistic_regression(y_tr_run, X_tr_run, initial_w, max_iter, gamma ,tol=1e-8)
                
                # Skip invalid runs
                if np.isnan(loss) or np.isinf(loss) or loss > 10:
                    print(f"Invalid loss ({loss:.4f}), skipping.")
                    continue

                # Evaluate all thresholds for this model
                for th in thresholds:
                    preds, _ = predict_with_threshold(X_proc_val, w, threshold=th)
                    f1 = f1_score(y_val, preds)
                    results.append((thr_nans, meth, th, f1))

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (thr_nans,meth, th)
                        print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

            except Exception as e:
                print(f"Error for method={meth},thr_nans={thr_nans}: {e}")
                continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[3], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(f"Best F1 = {best_f1:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, threshold={best_params[2]}")
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results


## S'ha de corretgir!!!! ( potser millor copiar la que tenia el jan i  començar de 0)
def grid_search_logistic_wheight(
    y_train, X_train,
    y_val, X_val,
    thershold_nans = [0.4, 0.3, 0.2],
    pos_weights=[1, 3, 5, 9],
    methods = ["undersample", "oversample","weighted"],
    lambdas=[1e-5, 1e-3, 1e-2, 1e-1],
    thresholds=[0.3, 0.5, 0.7],
    max_iter=10000,
    gamma=0.05
):
    """
    Grid search for weighted penalized logistic regression.
    Returns: best_params, best_f1, results_list
    """
    best_f1 = -1
    best_params = None
    results = []

    total = len(methods) * len(thershold_nans) * len(pos_weights) * len(lambdas) * len(thresholds)
    run = 0
    for thr_nans in thershold_nans:
        keep_mask = remove_nan_feat(X_train, threshold= thr_nans)
        X_clean_nan_tr = X_train[:,keep_mask]
        X_clean_nan_val = X_val[:,keep_mask]
        cat_mask, num_mask = detect_categorical_features(X_clean_nan_tr, max_unique=30)
        X_imp_tr,medians_tr = impute_numerical(X_clean_nan_tr, num_mask, medians=None, means=None, strategy='medians')
        X_imp_tr,modes_tr = impute_categorical(X_clean_nan_tr, cat_mask, modes=None)
        X_imp_val, _ = impute_numerical(X_clean_nan_val, num_mask, medians=medians_tr, strategy='medians')
        X_imp_val, _ = impute_categorical(X_clean_nan_val, cat_mask, modes=modes_tr)
        X_ohe_tr, categories = one_hot_encode(X_imp_tr, cat_mask)   
        X_ohe_val, _ = one_hot_encode(X_imp_val, cat_mask,categories = categories)
        mask = feature_target_correlation(X_ohe_tr, y_train, threshold_elimination=0.01)
        X_cleaned_tr = X_ohe_tr[:, mask]
        mask_correlated_variables = filter_colinear_features (X_cleaned_tr, y_train, 0.9)
        X_train = X_cleaned_tr[:, mask_correlated_variables]
        X_val = X_ohe_val[:, mask][:, mask_correlated_variables]
        X_train, means_tr, stds_tr= standardize_features(X_train, num_mask, means=None, stds=None)
        X_val, _, _ = standardize_features(X_val, num_mask, means=means_tr, stds=stds_tr)
        if np.isnan(X_train).any():
            print("Warning: X_train still contains NaNs! Impute them before training.")
        for pw in pos_weights:
            for lam in lambdas:
                for meth in methods: 
                    run += 1
                    print(f"\n=== Run {run}/{total//len(thresholds)} (method={meth},thr_nans = {thr_nans}, pos_weight={pw}, lambda_={lam}) ===")
                    try:
                        if meth != "weighted": 
                            X_train,y_train = balance_data(X_train,y_train,method = meth,random_state=None)
                            w,loss = logistic_regression(y_train, X_train, np.zeros(X_train.shape[1]), 10000 ,gamma,1e-8)
                        else: 
                            # Train model
                            loss, w = logistic_regression_weighted_gd(
                                y_train, X_train,
                                lambda_=lam, gamma=gamma,
                                pos_weight=pw, neg_weight=1.0,
                                max_iter=max_iter, verbose=False )

                        # Skip invalid runs
                        if np.isnan(loss) or np.isinf(loss) or loss > 10:
                            print(f"Invalid loss ({loss:.4f}), skipping.")
                            continue

                        # Evaluate all thresholds for this model
                        for th in thresholds:
                            preds, _ = predict_with_threshold(X_val, w, threshold=th)
                            f1 = f1_score(y_val, preds)
                            results.append((thr_nans, meth, pw, lam, th, f1))

                            if f1 > best_f1:
                                best_f1 = f1
                                best_params = (thr_nans,meth, pw, lam, th)
                                print(f"New best F1 = {best_f1:.4f} (threshold = {th:.2f})")

                    except Exception as e:
                        print(f"Error for method={meth},thr_nans={thr_nans}, pos_weight={pw}, lambda_={lam}: {e}")
                        continue

    # Sort results by F1 descending
    results.sort(key=lambda t: t[5], reverse=True)

    print("\n=== Grid Search Complete ===")
    if best_params:
        print(f"Best F1 = {best_f1:.4f} at thr_nans={best_params[0]},meth = {best_params[1]}, pos_weight={best_params[2]}, λ={best_params[3]}, threshold={best_params[4]}")
    else:
        print("No valid runs completed.")

    return best_params, best_f1, results
