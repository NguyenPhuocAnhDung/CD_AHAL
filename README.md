# \# CD-AHAL: Concept Drift-Aware Hybrid Active Learning for Network Intrusion Detection

# 

# CD-AHAL is a concept drift-aware Network Intrusion Detection System (NIDS) designed for dynamic IoT and heterogeneous network environments.  

# The framework combines a \*\*CNN–BiGRU–Attention\*\* backbone with \*\*uncertainty-aware active learning\*\* to maintain detection performance when traffic distributions evolve over time.

# 

# \---

# 

# \## Overview

# 

# Traditional NIDS models are often trained once and deployed as static classifiers. In real environments, this setup degrades quickly because:

# 

# \- network behavior changes over time

# \- new attack variants emerge continuously

# \- class imbalance causes rare attacks to be overlooked

# \- continuous retraining is expensive and impractical

# 

# CD-AHAL addresses these problems through a hybrid adaptive pipeline that:

# 

# \- detects drift early using predictive uncertainty

# \- triggers selective relabeling only when needed

# \- fine-tunes efficiently under a limited labeling budget

# \- preserves deployment realism with frozen preprocessing during streaming inference

# 

# \---

# 

# \## Key Contributions

# 

# \- \*\*Hybrid detection backbone\*\* using CNN, Bi-GRU, and Attention

# \- \*\*Concept drift detection\*\* via MC Dropout and entropy-based uncertainty

# \- \*\*Adaptive active learning loop\*\* with simulated Oracle labeling

# \- \*\*Two-phase detection strategy\*\* for both rapid screening and fine-grained identification

# \- \*\*Resource-aware online adaptation\*\* for practical edge and IoT environments

# 

# \---

# 

# \## Method Summary

# 

# \### 1. Frozen Preprocessing Pipeline

# Raw traffic from multiple sources is cleaned, aligned, and standardized into a shared feature space.  

# During online deployment, preprocessing parameters remain frozen to avoid leakage and better simulate real-world operation.

# 

# \### 2. CNN–BiGRU–Attention Backbone

# The main model combines:

# 

# \- \*\*1D-CNN\*\* for local feature extraction

# \- \*\*Bi-GRU\*\* for temporal dependency modeling

# \- \*\*Attention\*\* for focusing on informative signals and improving minority-class sensitivity

# 

# \### 3. Uncertainty Estimation with MC Dropout

# Instead of waiting for performance collapse, CD-AHAL monitors \*\*predictive entropy\*\* from stochastic forward passes.  

# A sudden entropy spike acts as an early warning for concept drift.

# 

# \### 4. Active Learning + Simulated Oracle

# Once drift is detected, the system:

# 

# \- selects the most uncertain samples

# \- requests labels from a simulated Oracle

# \- pseudo-labels very high-confidence samples

# \- fine-tunes incrementally instead of retraining from scratch

# 

# \---

# 

# \## Two-Phase Detection Strategy

# 

# \### Phase 1 — Rapid Screening

# Binary classification:

# 

# \- \*\*Normal (Benign)\*\*

# \- \*\*Attack\*\*

# 

# This stage prioritizes fast early detection and reduces downstream load.

# 

# \### Phase 2 — Fine-Grained Identification

# Multiclass classification assigns malicious traffic to specific attack categories such as:

# 

# \- DDoS

# \- PortScan

# \- Botnet

# \- WebAttack

# \- and other attack families defined in the label space

# 

# This hierarchical design balances real-time alerting and forensic usefulness.

# 

# \---

# 

# \## Multi-Source Dataset Design

# 

# \### Offline Training

# The initial model is trained on:

# 

# \- \*\*CICDDoS2017\*\*

# \- \*\*CICIoT2023\*\*

# 

# \### Composite Drift Evaluation

# The online drift stream is built from:

# 

# \- \*\*CSE-CIC-IDS2018\*\*

# \- \*\*CICDDoS2019\*\*

# \- \*\*CICDarknet2020\*\*

# \- \*\*L1-DoH-NonDoH\*\*

# 

# \### Unified Data Space

# After preprocessing:

# 

# \- offline training set: \*\*4,800,000 samples\*\*

# \- online drift stream: \*\*1,277,823 samples\*\*

# \- shared feature space: \*\*33 dimensions\*\*

# \- sequential aggregation: every \*\*10 consecutive records\*\*

# \- final online stream: \*\*127,488 sequences\*\*

# \- processed in \*\*249 batches\*\* of \*\*512 sequences\*\*

# 

# \---

# 

# \## Experimental Highlights

# 

# \### Phase 1 Binary Training

# CD-AHAL achieved strong offline detection performance:

# 

# \- \*\*Accuracy:\*\* 98.53%

# \- \*\*Precision:\*\* 98.90%

# \- \*\*Recall:\*\* 99.20%

# \- \*\*F1-score:\*\* 99.05%

# 

# \### Phase 2 Multiclass Evaluation

# CD-AHAL outperformed the baseline variants:

# 

# \- \*\*Accuracy:\*\* 77.99%

# \- \*\*Precision:\*\* 78.09%

# \- \*\*Recall:\*\* 77.99%

# \- \*\*F1-score:\*\* 77.33%

# 

# \### Drift-Aware Online Binary Evaluation

# CD-AHAL maintained competitive performance under streaming concept drift:

# 

# \- \*\*Accuracy:\*\* 90.76%

# \- \*\*Precision:\*\* 90.78%

# \- \*\*Recall:\*\* 90.76%

# \- \*\*F1-score:\*\* 90.77%

# 

# \### Efficiency

# Compared with continuous-update methods, CD-AHAL reduced update overhead substantially:

# 

# \- \*\*97 updates\*\*

# \- \*\*38.9% update rate\*\*

# \- about \*\*61% fewer retraining-related updates\*\* than full continuous updating

# \- average adaptation latency around \*\*769 ms\*\*

# 

# \---

# 

# \## Why CD-AHAL Matters

# 

# CD-AHAL is designed for settings where both \*\*security quality\*\* and \*\*operational efficiency\*\* matter.

# 

# It is especially suitable for:

# 

# \- IoT intrusion detection

# \- edge-deployed security pipelines

# \- dynamic streaming environments

# \- cost-sensitive adaptive learning scenarios

# \- research on concept drift and active learning in cybersecurity

# 

# \---

# 

# \## Core Ideas in One View

# 

# \- \*\*Static models fail under drift\*\*

# \- \*\*Error-based drift detection reacts too late\*\*

# \- \*\*MC Dropout + entropy provides earlier warning\*\*

# \- \*\*Active learning reduces labeling cost\*\*

# \- \*\*Hybrid CNN–BiGRU–Attention balances accuracy and efficiency\*\*

# \- \*\*Two-phase detection improves both alerting and analysis\*\*

# 

# \---

# 

# \## Minimal Tech Stack

# 

# \- Python 3.9

# \- TensorFlow 2.10

# \- Keras 2.10

# \- NumPy

# \- Pandas

# \- Scikit-learn

# 

# \---

# 

# \## Recommended Workflow

# 

# 1\. Prepare and align multi-source network datasets

# 2\. Build a unified offline training set

# 3\. Train the initial CNN–BiGRU–Attention backbone

# 4\. Freeze preprocessing for deployment

# 5\. Stream composite batches for drift evaluation

# 6\. Measure uncertainty with MC Dropout

# 7\. Trigger active relabeling when entropy spikes

# 8\. Fine-tune under a constrained labeling budget

# 9\. Evaluate binary and multiclass performance

# 10\. Compare against static and continuous-update baselines

# 

# \---

# 

# \## Accepted Papers

# 

# This repository is associated with the following accepted research outputs:

# 

# \### 1. FJCAI 2026 Conference Paper

# \*\*CD-AHAL: A Concept Drift-Aware Hybrid Active Learning Framework\*\*

# 

# \### 2. Journal of Computer Science and Cybernetics

# \*\*CD-AHAL: A Concept Drift-Aware Hybrid Active Learning Framework for Network Intrusion Detection\*\*  

# Accepted for publication in 2026.

# 

# \---

# 

# \## Citation

# 

# If you use this project in academic work, please cite the associated paper.

# 

# \### BibTeX

# ```bibtex

# @article{cdahal2026,

# &#x20; title   = {CD-AHAL: A Concept Drift-Aware Hybrid Active Learning Framework for Network Intrusion Detection},

# &#x20; author  = {Nguyen, Phuoc Anh Dung and Nguyen, Viet Hung and Tran, Van Thang and Hoang, Van Quy and Nguyen, Tat Thang and Le, Thien Huy and Bui, Huong},

# &#x20; journal = {Journal of Computer Science and Cybernetics},

# &#x20; year    = {2026},

# &#x20; note    = {Accepted}

# }

