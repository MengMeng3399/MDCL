# MDCL
# Multi-level Dual Contrastive Learning for Cloud API Cold-Start Recommendation
⭐ This code has been completely released ⭐

⭐ Overall framework of the MDCL model⭐ 
![framework](https://github.com/user-attachments/assets/f6e61ea3-ce02-4371-be7f-39bad37f7957)
Overall framework of the MDCL model.

⭐ The HGA dataset refers to our previous work:[ https://github.com/528Lab/CAData](https://github.com/528Lab/CAData)⭐ 

⭐ The PWA dataset refers to: [https://github.com/kkfletch/API-Dataset](https://github.com/kkfletch/API-Dataset)⭐ 

## Environment

- **Python**: 3.9.16  
- **PyTorch**: 2.0.1  
- Common dependencies: `numpy`, `pandas`, `scikit-learn`, `tqdm`

---

## Dataset Preparation

Download datasets from the official sources above and organize them as:

    data/
      HGA/    # files from CAData
      PWA/    # files from API-Dataset

---

## Workflow 
**Data processing**
    
    python data_processing.py

**Data splitting**
    
    python split_data.py

**Model training & evaluation**
    
    python MDCL.py

If your folder layout differs, adjust the data path arguments/constants;

---





