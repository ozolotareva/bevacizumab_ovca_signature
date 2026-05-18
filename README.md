# Transcriptome signature for the identification of bevacizumab responders in ovarian cancer

Cite: [preprint](https://arxiv.org/abs/2501.04869)

* [survival_in_UKE_and_DASL.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/survival_in_UKE_and_DASL.ipynb) - survival analysis of complete cohorts stratified by treatment.
* [discover_and_analyze_biclusters.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/discover_and_analyze_biclusters.ipynb) - identification of biclusters in the UKE dataset with UnPaSt, selection of predictive bicluster candidates presenting in DASL, Random Survival Forest and Cox regression analyses.  
* [heterogeneity.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/heterogneity.ipynb) - validation of biclusters identified in the UKE in the DASL and TCGA-OV data, identification of biclusters best matching known molecular subtypes.
* [consensusOV.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/consensusOV.ipynb) - classification of tumor expression profiles into known molecular subtypes using supervised consensusOV classifier.
* [subtypes_and_survival.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/subtypes_and_survival.ipynb) - analysis of known molecular sutbypes identified by consensusOV.
* [UKE_count_normalization_and_filtering.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/UKE_count_normalization_and_filtering.ipynb) - preprocessing of raw count data.
* [limma_DE.ipynb](https://github.com/ozolotareva/bevacizumab_ovca_signature/blob/main/limma_DE.ipynb) - differential expression analysis for microarray and RNA-seq data.
