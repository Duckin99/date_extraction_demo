import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_evaluation_image(tp, fp, fn, tn):
    # 1. คำนวณ Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # 2. เตรียม Data สำหรับ Confusion Matrix
    cm = np.array([[tp, fp], [fn, tn]])
    df_cm = pd.DataFrame(cm, index=['Actual: Stamp', 'Actual: No Stamp'], 
                         columns=['Pred: Stamp', 'Pred: No Stamp'])
    
    # 3. ตั้งค่า Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Reds', annot_kws={"size": 16}, cbar=False, ax=ax)
    
    # ปรับแต่งข้อความ
    plt.title(f'End-to-End Extraction Result\nAccuracy: {accuracy:.2%}', fontsize=16, pad=20)
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    
    # เพิ่มตาราง Metrics ด้านล่างรูป
    metrics_text = (
        f"Precision: {precision:.2%}\n"
        f"Recall:    {recall:.2%}\n"
        f"F1-Score:  {f1:.2%}"
    )
    plt.text(0, -0.5, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save และโชว์
    plt.tight_layout()
    plt.savefig('e2e_evaluation_report.png', dpi=300)
    print("Generate file: e2e_evaluation_report.png success!")
    plt.show()

# รันฟังก์ชันด้วยค่าของคุณ
generate_evaluation_image(tp=37, fp=2, fn=20, tn=25)