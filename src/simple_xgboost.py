import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier, plot_importance

import utils


def do_xgboost(df_path):
    # Load your data
    df = pd.read_csv(df_path)

    # Target encoding

    def target_encode(series, target):
        # Calculate mean target for each category
        means = df.groupby(series)[target].mean()
        # Map the means to the series
        return series.map(means)

    # Apply target encoding

    def apply_target_encoding(field, df):
        df[field] = target_encode(df[field], utils.FIELD_CLASSIFIER_TRUTH)

    apply_target_encoding(utils.FIELD_SENDER, df)
    utils.repeat_for_subject_and_body(apply_target_encoding, df)

    # Drop original columns and unwanted features
    X = df.drop(
        columns=[
            utils.FIELD_CLASSIFIER_TRUTH,
            utils.FIELD_HASH,
            # utils.FIELD_TAGGED_USEFUL,
            # utils.FIELD_TAGGED_USELESS,
        ]
    )
    y = df[utils.FIELD_CLASSIFIER_TRUTH]

    # Standardize numerical features
    scaler = StandardScaler()
    X[
        [
            utils.FIELD_SENDER_FREQ_MONTH,
            utils.FIELD_SENDER_FREQ_WEEK,
            utils.FIELD_SENDER_FREQ_DAY,
            utils.FIELD_HOUR,
            utils.FIELD_TAGGED_USEFUL,
            utils.FIELD_TAGGED_USELESS,
        ]
    ] = scaler.fit_transform(
        X[
            [
                utils.FIELD_SENDER_FREQ_MONTH,
                utils.FIELD_SENDER_FREQ_WEEK,
                utils.FIELD_SENDER_FREQ_DAY,
                utils.FIELD_HOUR,
                utils.FIELD_TAGGED_USEFUL,
                utils.FIELD_TAGGED_USELESS,
            ]
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # XGBoost Model
    model = XGBClassifier(eval_metric="logloss", scale_pos_weight=10)

    # Add manual weights
    # weights = (
    #     df["tagged_useful_alert"] * utils.MANUAL_TAG_WEIGHT
    #     + df["tagged_useless_alert"] * utils.MANUAL_TAG_WEIGHT
    # )

    # ,weight=weights.loc[X_train.index])
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Train model
    model.fit(X_train, y_train)  # ,sample_weight=weights.loc[X_train.index])
    return model, y_test, X_test, X_train


def show_accuracy(model, y_test, X_test):
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    return y_pred


def show_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def show_feature_importance(model, X_train):
    # Get feature importances
    importances = model.feature_importances_

    # Print feature importances
    for feature, importance in zip(X_train.columns, importances):
        print(f"Feature: {feature}, Importance: {importance}")

    # Plot feature importances
    plot_importance(model, importance_type="weight", title="Feature Importance")
    plt.show()


def main():
    model, y_test, X_test, X_train = do_xgboost(utils.METADATA_PATH)
    y_pred = show_accuracy(model, y_test, X_test)
    show_confusion_matrix(y_test, y_pred)
    show_feature_importance(model, X_train)


if __name__ == "__main__":
    main()
