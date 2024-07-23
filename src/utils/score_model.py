import os
import pandas as pd
import numpy as np

def calculate_model_score(model_performance, data_usage, baseline_accuracy=0.6324368170257361):
    
    # Calculate total data usage
    total_data_usage = sum(data_usage.values())
    num_languages = len(model_performance)

    # Accuracy calculation
    accuracy_score = (1 / num_languages) * (np.sum(lang_accuracy / baseline_accuracy for lang_accuracy in model_performance.values()))

    # Data usage calculation (percentage based normalization)
    k = 3000
    data_score = (total_data_usage / (total_data_usage + k)) 

    # Weights
    acc_weight = 0.6
    data_weight = 0.4

    # Calculate composite score
    composite_score = (acc_weight * accuracy_score) + (data_weight * (1 / data_score))

    # Create dataframe to store model info and save
    new_data = {
        'Data Usage': [total_data_usage],
        'Accuracy Distribution': [list(model_performance.values())],
        'Composite Score': [composite_score],
        'Accuracy Score': [accuracy_score],
        'Data Score': [data_score]
    }
    file_path = "src/model_performance/model_performance.csv"
    df = pd.DataFrame(new_data)
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
    
    return composite_score, accuracy_score, data_score