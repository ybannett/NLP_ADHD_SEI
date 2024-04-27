# Project Summary:

This study aims to evaluate the accuracy of a large language model in identifying documentation of ADHD medication side effects in various clinical encounter notes, potentially enhancing quality measurement and patient outcomes. It complements our previous study [NLP_ADHD_PTBM](https://github.com/ybannett/NLP_ADHD_PTBM)
 that focused on clinician adherence to AAP guidelines in recommending non-pharmacological behavioral treatment for young children with ADHD, in which we demonstrated the use of large language models in the evaluation of quality-of-care for children with ADHD [1]. These two studies are the first, to our knowledge, that provide objective support for the successful use of artificial intelligence (AI) to provide a comprehensive evaluation of ADHD management, a prevalent neurobehavioral condition that is predominantly managed in primary care.

# Dataset:

The dataset used consists of structured and unstructured (free text) EHR data (2015-2022) from 11 community primary care practices. It includes all office, telehealth, and telephone encounters related to ADHD for patients aged 6-11 years. To examine medication side effects documentation, the study focused on patients with more than two ADHD medication encounters, involving prescriptions of stimulants or non-stimulants by primary care physicians.

# Repository Structure:

The repo consists of the src folder, which has two files listed below:

1. text_preprocessing_training: This Python file contains the necessary code for processing text data from the notes, extracting feature embeddings from LLaMA-13b, and training the network using feature embeddings.

2. downstream_analysis:This notebook contains the codes used for downstream analysis and figures.


# Citations:

 1. Pillai M, Posada J, Gardner RM, et al. Measuring quality-of-care in treatment of young children with attention-deficit/hyperactivity disorder using pre-trained language models. Journal of the American Medical Informatics Association. 2024;doi:10.1093/jamia/ocae001
 2. Wolraich ML, Hagan JF, Jr., Allan C, et al. Clinical Practice Guideline for the Diagnosis, Evaluation, and Treatment of Attention-Deficit/Hyperactivity Disorder in Children and Adolescents. Pediatrics. Oct 2019;144(4)doi:10.1542/peds.2019-2528
 3. Epstein JN, Kelleher KJ, Baum R, et al. Variability in ADHD care in community-based pediatrics. Pediatrics. Dec 2014;134(6):1136-43. doi:10.1542/peds.2014-1500
 4. Zima BT, Murphy JM, Scholle SH, et al. National quality measures for child mental health care: background, progress, and next steps. Pediatrics. Mar 2013;131 Suppl 1:S38-49. doi:10.1542/peds.2012-1427e
 5. Epstein JN, Kelleher KJ, Baum R, et al. Specific Components of Pediatricians’ Medication-Related Care Predict Attention-Deficit/Hyperactivity Disorder Symptom Improvement. Journal of the American Academy of Child and Adolescent Psychiatry. Jun 2017;56(6):483-490.e1. doi:10.1016/j.jaac.2017.03.014
