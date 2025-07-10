# Understanding the Effect of Importance Weighting in Deep Learning under Dataset Shift

This repository contains the full source code and experiments for our study on how **importance weighting** affects deep learning performance when facing **dataset shift**, particularly in classification tasks.

## 📄 Abstract

Importance weighting is a standard technique to address dataset shift by correcting for discrepancies between training and test distributions. While effective for simple models, its behavior under modern deep learning remains poorly understood. In this study, we systematically investigate when and why importance weighting helps or fails in deep neural networks. Using both synthetic and real-world datasets (e.g., CIFAR-10), we analyze generalization, gradient behaviors, and the effects of reweighting in various shift scenarios.

## 📂 Repository Structure

```
.
├── CIFAR10_binary_classifier.py     # Binary classification under dataset shift
├── CIFAR10_extension.py            # Extension for multiclass experiments
├── CIFAR10_visualization.ipynb     # Analysis and plots on CIFAR-10
├── Synthetic_Data.ipynb            # Synthetic data experiments
└── README.md                       # Project overview (this file)
```

## 📊 Key Features

- Experiments on both **synthetic data** and **CIFAR-10**.
- Comparative analysis of:
  - Empirical Risk Minimization (ERM)
  - Importance Weighted ERM (IW-ERM)
- Reproduction of common dataset shift scenarios:
  - Label shift
  - Covariate shift
- Visualization of learned decision boundaries and gradient dynamics.

## 🧪 Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, Matplotlib, scikit-learn, seaborn
- Jupyter Notebook (for exploratory analysis)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 Running Experiments

To train a binary classifier on CIFAR-10 with dataset shift:

```bash
python CIFAR10_binary_classifier.py
```

To explore synthetic data experiments:

```bash
jupyter notebook Synthetic_Data.ipynb
```

## 📈 Results Highlights

- Importance weighting improves generalization **only under certain shift types** and **for shallow networks**.
- In deep networks, importance weights are often overridden by optimization dynamics (e.g., gradient saturation).
- Visualization reveals decision boundaries become unstable under extreme weighting.

## 📚 Citation

If you use this code or find it helpful in your work, please cite:

```bibtex
@misc{Thien Nhan Vo,
  title={Understanding the Effect of Importance Weighting in Deep Learning under Dataset Shift},
  author={Thien Nhan Vo},
  year={2025},
  eprint={arXiv:2505.03617},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## 🔗 License

This project is licensed under the MIT License. See `LICENSE` for details.

## 🙋 Contact

For questions or collaborations, please reach out at:  
📧 thiennhan.math@gmail.com
