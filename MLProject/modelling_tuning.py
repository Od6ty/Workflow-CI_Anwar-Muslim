import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup MLflow Tracking (Lokal)
# Saat di GitHub Actions, ini akan membuat folder ./mlruns
# Jika dijalankan via mlflow run, tracking URI dan experiment sudah di-set oleh mlflow run
# Hanya set jika belum di-set (untuk local testing)
if os.getenv("MLFLOW_TRACKING_URI") is None:
    mlflow.set_tracking_uri("file:./mlruns")
if os.getenv("MLFLOW_EXPERIMENT_NAME") is None:
    mlflow.set_experiment("Eksperimen_Taxi_Skilled_Anwar")

def load_data(folder_path):
    print(f"[INFO] Loading data from {folder_path}...")
    train_path = os.path.join(folder_path, "train.csv")
    test_path = os.path.join(folder_path, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File {train_path} tidak ditemukan!")
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Pisahkan Fitur dan Target
    # Pastikan nama target sesuai dengan preprocessing ('fare_amount')
    target_col = 'fare_amount'
    
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    return X_train, y_train, X_test, y_test

def main():
    # Load Data (Pastikan folder ini ada di sebelah script ini)
    # Di GitHub Actions, struktur folder akan sama seperti yang kita buat
    X_train, y_train, X_test, y_test = load_data("taxi_tripdata_preprocessing")

    # Definisi Hyperparameter Space untuk Tuning
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [5, 10, None]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    # Grid Search (CV=3 agar cepat)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        verbose=1
    )

    # Start Run (Manual Logging untuk Skilled)
    # Jika sudah ada active run (dari mlflow run), gunakan nested run
    print("[INFO] Memulai MLflow Run...")
    active_run = mlflow.active_run()
    if active_run is None:
        # Tidak ada active run, buat run baru
        run_context = mlflow.start_run(run_name="Hyperparameter_Tuning_Skilled")
    else:
        # Ada active run dari mlflow run, buat nested run
        run_context = mlflow.start_run(run_name="Hyperparameter_Tuning_Skilled", nested=True)
    
    with run_context:
        
        print("[INFO] Memulai Grid Search...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"[RESULT] Best Params: {best_params}")
        
        # Evaluasi Model Terbaik
        predictions = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print(f"[RESULT] Metrics -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        # --- LOGGING MANUAL (Syarat Skilled) ---
        # 1. Log Parameters
        mlflow.log_params(best_params)
        
        # 2. Log Metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # 3. Log Model
        # Log model agar tersimpan sebagai artefak
        mlflow.sklearn.log_model(best_model, "model_taxi_anwar")
        
        print("[INFO] Run selesai. Model dan metrik tersimpan di MLflow.")

if __name__ == "__main__":
    main()