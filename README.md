# PKGCL
PKGCL is designed to predict drug response in cancer cell lines. In this framework, gene-expression data are transformed into 50 Hallmark GSVA pathway features, and a Hallmark pathway graph is constructed to model pathway relationships. An MLP and a graph attention network (GAT) learn complementary pathway representations, which are fused by a gating mechanism to obtain a cell representation. This cell representation is combined with a drug representation derived from 2048-dimensional drug features to classify each cancer cell line-drug pair as sensitive or resistant. The model is trained with binary classification and uncertainty-aware adaptive interaction contrastive learning, which uses cell similarity, drug similarity, and MC Dropout-based predictive confidence. After model training, Integrated Gradients is used to quantify the importance of each Hallmark pathway for prediction.

# Requirements
Python 3.12.3 and R 4.5.1.
Main Python packages: numpy 2.5.4, scikit-learn 1.9.1, PyTorch 2.6.0, and PyTorch Geometric 2.8.0.
R packages: GSVA 2.4.9 and BiocParallel 1.44.0.

# Data preparation
In this study, we utilized cancer cell line gene-expression data, drug-response data, and drug fingerprint data. Drug response was measured by LN_IC50 values. Gene-expression data were used as cell inputs. Drug fingerprint data were used as drug inputs. The 50 Hallmark gene sets in hallmark.gmt were included for pathway analysis.

# Usage
PKGCL is a framework with five phases. Firstly, please use prepare_data.py in the phase1 folder to prepare cell expression data, drug features, and drug-response data. After that, hallmark_gsva.py in the phase2 folder is used to calculate Hallmark GSVA pathway scores. cancer_drug_audit.py and finalize_labels.py in the phase3 folder are
then used to generate drug-response thresholds and sensitivity labels. Next, build_pathway_graph.py in the phase4 folder is used to construct the Hallmark pathway graph. Finally, prepare_drug_holdout_split.py and PKGCL.py in the phase5 folder are used for drug-holdout evaluation and drug-response prediction. IG_analysis.py can be used to calculate Hallmark pathway importance.

Before using, please check the cancer code and file paths in the scripts.
