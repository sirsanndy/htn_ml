from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from models.extract_cv_results import extract_cv_results

def dummy_model(X_train_clean, y_train, X_valid_clean):
    model = DummyClassifier(strategy="stratified", random_state=123)
    param_grid = {}

    model.fit(X_train_clean, y_train)
    y_pred = model.predict(X_valid_clean)
    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring="f1_weighted", return_train_score=True)

    # Fit GridSearchCV
    grid_search.fit(X_train_clean, y_train)

    # Assign the fitted GridSearchCV object to reg_base
    reg_base = grid_search
    # Validate the CV Score (JUST RUN THE CODE)
    train_base, valid_base, best_base, best_param_base = extract_cv_results(reg_base)

    print(f'Train score - Baseline model: {train_base*100:.2f}%')
    print(f'Valid score - Baseline model: {valid_base*100:.2f}%')
    print(f'Best score - Baseline model: {best_base*100:.2f}%')
    print(f'Best Params - Baseline model: {best_param_base}')