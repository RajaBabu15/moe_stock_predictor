# Mixture-of-Experts (MoE) Stock Price Predictor

This project implements an advanced Mixture-of-Experts (MoE) model designed for stock price forecasting. It leverages modern deep learning architectures and techniques to capture complex market dynamics.

## Features

*   **Mixture-of-Experts (MoE) Core:** Dynamically routes input sequences to specialized "expert" networks.
*   **Diverse Expert Architectures:** Supports multiple expert types within a single model:
    *   LSTM / GRU (Recurrent Networks)
    *   Transformer Encoder Blocks (Attention-based)
    *   Temporal Convolutional Networks (TCN)
    *   *(Placeholder for NBEATS/NHiTS integration)*
*   **Advanced Gating Mechanisms:** Options for selecting experts:
    *   Simple LSTM / Dense Gates
    *   Context-Aware Gating (using temporal and feature context)
    *   Attention-MHA Gating (BiLSTM + Multi-Head Attention)
*   **Enhanced Feature Engineering:**
    *   Standard Technical Indicators (via `ta` library)
    *   Volatility Measures (Log Returns, Realized Volatility)
    *   Range and Average True Range (ATR)
    *   Frequency Domain Features (FFT)
    *   Fourier Time Features (capturing seasonality)
*   **Sophisticated Training:**
    *   Auxiliary Load Balancing Loss (encourages even expert usage)
    *   Optional Directional Loss (penalizes incorrect trend direction prediction)
    *   Mixed Precision Training support (for speedup on compatible GPUs)
    *   Quantization-Aware Training (QAT) support (for deployment optimization)
    *   Callbacks: Early Stopping, Model Checkpointing, TensorBoard logging
*   **Uncertainty Estimation:** Monte Carlo (MC) Dropout via a vectorized `ProbabilisticMoE` wrapper for prediction intervals.
*   **Financial Metrics:** Calculates Sharpe Ratio, Sortino Ratio, Max Drawdown, and Calmar Ratio for backtesting evaluation.
*   **Deployment Ready:**
    *   ONNX Export functionality.
    *   Dependency management via `requirements.txt` and `environment.yml`.
*   **Hyperparameter Tuning:** Integrated support for `keras-tuner`.
*   **Testing & CI:** Includes basic unit tests and a sample GitHub Actions workflow.
*   **Interactive Dashboard:** A simple `gradio` app for making predictions.
*   **Modular Structure:** Code organized into logical packages and modules.

## Key Concepts & Formulas

* **Mixture of Experts (MoE):**
  * A gating network $G(x)$ produces weights $w_i$ for $N$ experts $E_i(x)$.
  * Final Output:
    $$
    y = \sum_{i=1}^{N} w_i \cdot E_i(x)
    $$
    with the constraint:
    $$
    \sum_{i=1}^{N} w_i = 1.
    $$

* **Load Balancing Loss (Auxiliary):** Aims to distribute the input data evenly among experts. Often calculated using the Coefficient of Variation squared ($CV^2$) of the average gating probabilities per expert over a batch.
  $$
  L_{LB} = \alpha \cdot N^2 \cdot \frac{\text{Var}(P_i)}{\text{Mean}(P_i)^2}
  $$
  where $P_i$ is the average probability assigned to expert $i$ over the batch, $N$ is the number of experts, and $\alpha$ is a scaling weight.

* **Directional Loss (Optional Primary Loss Component):** Penalizes the model when the predicted direction of price *change* opposes the true direction.
  $$
  L_{Dir} = \text{MSE}(y_{\text{true}}, y_{\text{pred}}) + \beta \cdot \text{mean}\Bigl(\text{ReLU}\Bigl(1 - \text{sign}(\Delta y_{\text{true}}) \cdot \text{sign}(\Delta y_{\text{pred}})\Bigr)\Bigr)
  $$
  where $\Delta y$ represents the price change and $\beta$ is a weight.  
  *Note: Requires careful implementation based on prediction target (price vs. return) and output sequence length.*

* **MC Dropout:** Estimates model uncertainty by running inference multiple times with dropout layers active, then analyzing the distribution of predictions.

* **Sharpe Ratio:** Measures risk-adjusted return.
  $$
  \text{Sharpe} = \frac{R_p - R_f}{\sigma_p} \approx \frac{\text{mean}(\text{returns})}{\text{std}(\text{returns})} \times \sqrt{252}
  $$
  (assuming risk-free rate $R_f \approx 0$).

* **Sortino Ratio:** Similar to Sharpe, but only considers downside volatility.
  $$
  \text{Sortino} = \frac{R_p - R_f}{\sigma_d} \approx \frac{\text{mean}(\text{returns})}{\text{std}(\text{downside returns})} \times \sqrt{252}
  $$

* **Max Drawdown:** Largest peak-to-trough decline in portfolio value.

* **Calmar Ratio:** Annualized return divided by the absolute value of the maximum drawdown.

## Project Structure

```
moe_stock_predictor/
├── config/             # Configuration file
├── data/               # Input data
├── moe_model/          # Core model package (data, features, components, builder, etc.)
├── tests/              # Unit tests
├── utils/              # Utility functions (metrics, plotting, export)
├── app.py              # Gradio dashboard
├── train.py            # Main training script
├── predict.py          # Prediction script
├── tune.py             # Hyperparameter tuning script
├── .github/workflows/  # CI workflow example
├── .gitignore
├── environment.yml     # Conda environment
├── requirements.txt    # Pip requirements
└── README.md           # This file
```

## Setup

**1. Clone the Repository:**
```bash
git clone https://github.com/RajaBabu15/moe_stock_predictor
cd moe_stock_predictor
```

**2. System Dependencies (TA-Lib):**
The `TA-Lib` Python library requires the underlying TA-Lib C library to be installed first.

*   **Ubuntu/Debian:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential libta-lib-dev
    ```
*   **macOS (using Homebrew):**
    ```bash
    brew install ta-lib
    ```
*   **Windows:** Download `ta-lib-0.4.0-msvc.zip` from [SourceForge](https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/) or find pre-compiled wheels (e.g., from [UCI](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)). Follow specific instructions for setting up the C library on Windows.

**3. Create Environment & Install Packages:**

*   **Using Conda (Recommended):**
    ```bash
    conda env create -f environment.yml
    conda activate moe_stock_env
    # Install the TA-Lib Python wrapper *after* activating env
    pip install TA-Lib
    # Install any remaining pip-only dependencies if needed
    # pip install -r requirements.txt # Might double-install, use carefully
    ```

*   **Using Pip & Venv:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    # Install the TA-Lib Python wrapper *after* installing system deps
    pip install TA-Lib
    ```

**4. Prepare Data:**
Place your stock data CSV file (with columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`) in the `data/` directory. Update `config/config.py` if your filename or target column differs from `sample_data.csv` / `Close`. The script will create dummy data if the file is not found.

## Running the Project

*   **Train the Model:**
    ```bash
    python train.py
    ```
    This will:
    *   Load data and configuration.
    *   Preprocess data and create features.
    *   Build the MoE model based on `config.py`.
    *   Train the model, using callbacks (EarlyStopping, Checkpointing, TensorBoard).
    *   Evaluate the best model on the test set.
    *   Plot training history and predictions.
    *   (Optionally) Export the trained model to ONNX format.
    *   Checkpoints are saved in `model_checkpoints/`.
    *   TensorBoard logs are saved in `logs/fit/`. View with: `tensorboard --logdir logs/fit`

*   **Hyperparameter Tuning:**
    ```bash
    python tune.py
    ```
    This uses KerasTuner to search for optimal hyperparameters defined in `tune.py`. Results and the best tuned model are saved in the `hyperparam_tuning/` directory.

*   **Make Predictions:**
    *(Note: Ensure scalers were saved during training or implement logic to load/refit them in `predict.py`)*
    ```bash
    # First, ensure scalers are saved after training (add to train.py if needed):
    # import joblib
    # joblib.dump(scaler_target, 'scaler_target.joblib')
    # joblib.dump(scaler_features, 'scaler_features.joblib')

    # Then run prediction:
    python predict.py
    ```
    This loads the *best trained model* from the checkpoint and predicts the next step based on the latest data in the CSV.

*   **Run the Gradio Dashboard:**
    *(Note: Requires saved model and scalers)*
    ```bash
    python app.py
    ```
    This launches a web interface (usually at `http://127.0.0.1:7860`) where you can input data (or use the sample) to get predictions.

*   **Run Tests:**
    ```bash
    pytest tests/
    ```

## Output

*   **Console:** Logs progress, evaluation metrics (MSE, MAE, Load Balancing Loss), financial metrics (Sharpe, Sortino, etc.), and final results.
*   **Plots:** Matplotlib windows showing Training History and Actual vs. Predicted prices (with optional MC Dropout uncertainty bands).
*   **`model_checkpoints/`:** Saved Keras model file (`.keras`) of the best performing model during training.
*   **`logs/fit/`:** TensorBoard logs for visualizing training progress, metrics, and potentially model graph and histograms.
*   **`model_onnx/`:** Exported ONNX model file (if enabled).
*   **`hyperparam_tuning/`:** KerasTuner logs and saved best model from tuning.
*   **(Requires Manual Save):** `scaler_target.joblib`, `scaler_features.joblib` if saved after training.

## Future Extensions

*   Implement NBEATS/NHiTS/other advanced time-series experts.
*   Implement more sophisticated gating mechanisms (e.g., top-k routing, Expert Choice).
*   Integrate attention-based aggregation for expert outputs.
*   Add more comprehensive unit and integration tests.
*   Develop more advanced backtesting strategies within `utils/metrics.py`.
*   Optimize data loading for very large datasets (e.g., using `tf.data` more extensively with generators or TFRecords).
