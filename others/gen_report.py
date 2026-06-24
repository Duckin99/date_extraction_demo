import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_evaluation_image(tp, fp, fn, tn):
    # 1. Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # 2. Confusion Matrix (Truth on Y, Pred on X)
    cm = np.array([[tp, fn], [fp, tn]]) # Row=Truth, Col=Pred
    df_cm = pd.DataFrame(cm, 
                         index=['Actual: Stamp', 'Actual: No Stamp'], 
                         columns=['Pred: Stamp', 'Pred: No Stamp'])
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, cbar=False, ax=ax)
    
    plt.title(f'End-to-End Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=16, pad=20)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    
    # Add Metrics Text
    metrics_text = (
        f"Precision: {precision:.2%}\n"
        f"Recall:    {recall:.2%}\n"
        f"F1-Score:  {f1:.2%}"
    )
    plt.text(0, -0.4, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('res/e2e_evaluation_report.png', dpi=300)
    plt.show()

os.makedirs('res', exist_ok=True)
generate_evaluation_image(tp=37, fp=2, fn=20, tn=25)