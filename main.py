from src.train import train_model
from src.evaluate import evaluate_model

from src.data.panx_loader import PANX_dataloader

def main():

    languages = ["en", "es", "zh"]
    nrows = 200

    # -------------- Load data --------------
    print("Loading data...")
    dataloader = PANX_dataloader(langs=languages, nrows=nrows)
    df_train, df_val, df_test = dataloader.load_training_data()

    # -------------- Training --------------
    model_params = {
            'learning_rate': 3e-3,
            'epochs': 5,
            'batch_size': 2,
        }
    model_path = 'src/models/pretrained/test_model.pth'

    train_model(df_train, df_val, df_test, model_params, model_path)

    # -------------- Evaluation --------------
    evaluation_metrics = evaluate_model(model_path, df_test)
    print("Evaluation results:\n", evaluation_metrics)

if __name__ == "__main__":
    main()
