# Product Pricer

This project aims to build and fine-tune a large language model to predict the price of a product based on its description.

## Setup

1.  **Install PyTorch with CUDA:**

    Run the following command to install the correct version of PyTorch for your CUDA environment:

    ```bash
    ./install_pytorch.sh
    ```

2.  **Install other dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use the `src/main.py` script to perform various actions.

### Data Curation

To curate the data, run:

```bash
python src/main.py curate_data
```

### Show Data Point

To show a random data point, run:

```bash
python src/main.py show_data_point
```

### Pricing Models

To run the different pricing models, use the following commands:

-   **Random Pricer:**
    ```bash
    python src/main.py random_pricer
    ```
-   **Average Pricer:**
    ```bash
    python src/main.py average_pricer
    ```
-   **Frontier Model:**
    ```bash
    python src/main.py frontier --model <model_name>
    ```
-   **Frontier Fine-tuning:**
    ```bash
    python src/main.py frontier_finetuning
    ```
-   **Open Source Prediction:**
    ```bash
    python src/main.py open_source_prediction
    ```
-   **Open Source Fine-tuning Prediction:**
    ```bash
    python src/main.py open_source_finetuning_prediction
    ```
-   **QLoRA Fine-tuning:**
    ```bash
    python src/main.py qlora_finetuning
    ```
