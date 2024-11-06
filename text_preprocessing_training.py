# Importing essential libraries for data processing, deep learning, and NLP models.
import os
import shutil
import pandas as pd
import csv
import re
import numpy as np
import random as python_random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import torch
import matplotlib.pyplot as plt
import timeit
import nltk
import string
import math
from official.nlp import optimization  # to create AdamW optimizer
from tensorflow.keras.optimizers.legacy import SGD, Adam, Adamax
from tensorflow.keras.optimizers.experimental import AdamW
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Model,model_from_json,load_model
from tensorflow.keras import layers
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split,StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoTokenizer, BioGptModel,BioGptTokenizer,RobertaTokenizer,RobertaModel,BertModel,BertTokenizer,LlamaTokenizer,LlamaModel
from tensorflow.keras.optimizers.legacy import SGD, Adam, Adamax
from nltk.corpus import stopwords


tf.get_logger().setLevel('ERROR')

from transformers import TFBertForSequenceClassification, glue_convert_examples_to_features, TFDistilBertForSequenceClassification, TFDistilBertModel
from transformers import BertTokenizer, DistilBertTokenizer

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)

#Processes all txt files in a given directory path and extracts relevant information into a structured DataFrame.
def extractOriginalText(input_path):
    
    colnames=['file','note_des','length','mrn','visit_no']
    df_notes = pd.DataFrame(columns=colnames)
    results = [(f,os.path.join(dp, f)) for dp, dn, filenames in os.walk(input_path) for f in filenames if os.path.splitext(f)[1] == '.txt']
   
    for i in range(len(results)):
        filename = results[i][0]  
        row = pd.read_csv(results[i][1], sep="\t", quoting=csv.QUOTE_NONE).iloc[2][0]
        lenrow=len(row.split())
        mrn=re.findall(r"(\d+)_",filename)[0]
        visit_no= results[i][0][-5]
        new_row = {'file':filename, 'note_des':row, 'length':lenrow, 'mrn':mrn, 'visit_no':visit_no}
        df_notes=df_notes.append(new_row,ignore_index=True)
        
    return df_notes

#Process xmi files from a given directory to extract specific annotation tags 
def extractXMIAnnotation(input_path):
    # %% Write Patterns
    id_match = re.compile("(?<=xmi:id=\")\d{,5}(?=\")")
    begin_match = re.compile("(?<=begin=\")\d{,6}(?=\")")
    end_match = re.compile("(?<=end=\")\d{,6}(?=\")")
    tag_match = re.compile("(?<=semanticTag=\")\w*(?=\")")

    results = [(f.replace(".xmi",".txt"),os.path.join(dp, f)) for dp, dn, filenames in os.walk(input_path) for f in filenames if os.path.splitext(f)[1] == '.xmi']
    rows = []
    for i in range(len(results)):
        file = open(results[i][1], "r+")
        lines = file.readlines()
        lines = lines[0].split('><')
        
        extract = [x for x in lines if re.search("semanticTag", x)]
        
        extract = [(id_match.findall(x),
                    begin_match.findall(x),
                    end_match.findall(x),
                    tag_match.findall(x))
                   for x in extract]
        
        unique_tags = sorted(list(set([x[3][0] for x in extract])))
        
        file_name = re.findall("\\A.*(?=\.)",os.path.basename(results[i][0]))[0].split("-")

        row = pd.DataFrame({'xmi': [results[i][1]],
                            'file': [results[i][0]],
                            'anon_id': [file_name[0]]})

        for tag in unique_tags:
            row.loc[:, tag] = 1
            
        rows.append(row)

    full = pd.concat(rows).fillna(0)
    
    if 'BN_social' not in full:
        full['BN_social']=0
    if '_community' not in full:
        full['_community']=0
    if 'SE_other' not in full:
        full['SE_other']=0  
    if 'BN_personal' not in full:
        full['BN_personal']=0 
    if 'SE_psych_N' not in full:
        full['SE_psych_N']=0        
    if 'BN_other' not in full:
        full['BN_other']=0 
    if 'BN_academic' not in full:
        full['BN_academic']=0
    if 'SE_other_N' not in full:
        full['SE_other_N']=0
        
    full['SE_present'] = np.where((full['SE_physio'] == 1) |(full['SE_psych'] == 1)|(full['SE_other'] == 1), 1, 0)

    full['SE_absent'] = np.where((full['SE_physio_N'] == 1) |(full['SE_psych_N'] == 1)|(full['SE_other_N'] == 1), 1, 0)

    full['SE_inquiry'] = np.where((full['SE_present'] == 1) |(full['SE_absent'] == 1), 1, 0)
    
    full['BN_any'] = np.where((full['BN_symptoms'] == 1) |(full['BN_academic'] == 1)|(full['BN_personal'] == 1)|(full['BN_social'] == 1) |(full['_community'] == 1)|(full['BN_other'] == 1), 1, 0)

    return full

#Process xmi files to extract the location of annotations.
def SE_begin_end(input_path):
    # %% Write Patterns
    
    id_match = re.compile("(?<=xmi:id=\")\d{,5}(?=\")")
    begin_match = re.compile("(?<=begin=\")\d{,6}(?=\")")
    end_match = re.compile("(?<=end=\")\d{,6}(?=\")")
    tag_match = re.compile("(?<=semanticTag=\")\w*(?=\")")

    results = [(f.replace(".xmi",".txt"),os.path.join(dp, f)) for dp, dn, filenames in os.walk(input_path) for f in filenames if os.path.splitext(f)[1] == '.xmi']
    rows = []
    
    collist=['file','tag','begin','end']
    df=pd.DataFrame(columns=collist, index=None)
    for i in range(len(results)):
        file = open(results[i][1], "r+")
        lines = file.readlines()
        lines = lines[0].split('><')
        
        extract = [x for x in lines if re.search("semanticTag", x)]
    
        extract = [(id_match.findall(x),
                    begin_match.findall(x),
                    end_match.findall(x),
                    tag_match.findall(x))
                   for x in extract]
        file_name = re.findall("\\A.*(?=\.)",os.path.basename(results[i][0]))[0].split("-")
        for i in extract:
            new_row={'file':file_name[0], 'tag':i[3][0], 'begin':i[1][0], 'end':i[2][0]}
            df=df.append(new_row,ignore_index=True)
            
    return df
    
#Importing and consolidating data from multiple rounds of text notes and annotations.
originalTextData1 = extractOriginalText("/Original_notes/Round01/")
originalTextData2 = extractOriginalText("/Original_notes/Round02/")
originalTextData3 = extractOriginalText("/Original_notes/Round03/")
originalTextData4 = extractOriginalText("/Original_notes/Round04/")
originalTextData5 = extractOriginalText("/Original_notes/first010/")
originalTextData6 = extractOriginalText("/Original_notes/telephone/")
originalTextData=pd.concat([originalTextData1,originalTextData2,originalTextData3,originalTextData4,originalTextData5,originalTextData6], ignore_index=True)

annotatedXMIs1 = extractXMIAnnotation("/Annotated_notes/Round_1")
annotatedXMIs2 = extractXMIAnnotation("/Annotated_notes/Round_2")
annotatedXMIs3 = extractXMIAnnotation("/Annotated_notes/Round_3")
annotatedXMIs4 = extractXMIAnnotation("/Annotated_notes/Round_4")
annotatedXMIs5 = extractXMIAnnotation("/Annotated_notes/first_10")
annotatedXMIs6 = extractXMIAnnotation("/Annotated_notes/telephone")
annotatedXMIs=pd.concat([annotatedXMIs1,annotatedXMIs2,annotatedXMIs3,annotatedXMIs4,annotatedXMIs5,annotatedXMIs6], ignore_index=True)

df_annot_begin_end1 = SE_begin_end("/Annotated_notes/Round_1")
df_annot_begin_end2 = SE_begin_end("/Annotated_notes/Round_2")
df_annot_begin_end3 = SE_begin_end("/Annotated_notes/Round_3")
df_annot_begin_end4 = SE_begin_end("/Annotated_notes/Round_4")
df_annot_begin_end5 = SE_begin_end("/Annotated_notes/first_10")
df_annot_begin_end6 = SE_begin_end("/Annotated_notes/telephone")
df_annot_begin_end = pd.concat([df_annot_begin_end1,df_annot_begin_end2,df_annot_begin_end3,df_annot_begin_end4,df_annot_begin_end5,df_annot_begin_end6], ignore_index=True)
df_annot_begin_end.to_csv('df_annot_begin_end.csv')

# Filtering specific tags to create a subset DataFrame.
df_annot_subset=df_annot_begin_end[(df_annot_begin_end['tag']=='SE_other_N') | 
                                   (df_annot_begin_end['tag']=='SE_physio_N') |
                                   (df_annot_begin_end['tag']=='SE_psych_N') | 
                                   (df_annot_begin_end['tag']=='SE_other') | 
                                   (df_annot_begin_end['tag']=='SE_physio') |
                                   (df_annot_begin_end['tag']=='SE_psych')]
annotatedXMIs['SE_inquiry'].value_counts()

# Merging `originalTextData` and `annotatedXMIs` DataFrames.
data = originalTextData.merge(annotatedXMIs, on = "file", how = "right")

# Converting note to lowercase.
data['note_des_lower'] = data['note_des'].apply(lambda x: x.lower())

def extract_section_contents_single_report(report, sections_found, sections_list=None, new_line=True,
                                           delete_empty_sections=True):
    
    # this function delete all the sections titles found in an
    # attempt to clean the text an reduce the false positives
    # if new_line is equal to true the sections are appended using a new line character
    # if newline is false the sections are appended using a space
    # Sections_list: is a list with the specific sections to remove, by default None

    # iterate over all the found sections in the given report
    report_cleaned = []
    if sections_list is None:
        
        for i in range(len(sections_found.index)):
            span_section = sections_found.loc[i, ["start_section", "end_section"]]
            print(span_section)
            if (span_section[1] == span_section[0]) and delete_empty_sections:
                continue
            else:
                report_cleaned.append(report[span_section[0]:span_section[1]])
    else:
        indices = sections_found['section_name'].isin(sections_list)
        span_sections = sections_found.loc[indices, ["start_section", "end_section"]]
        for i in range(span_sections.shape[0]):
            if (span_sections.iloc[i, 0] == span_sections.iloc[i, 1]) and delete_empty_sections:
                continue
            else:
                report_cleaned.append([report[span_sections.iloc[i, 0]:span_sections.iloc[i, 1]]])
                
    return report_cleaned


def find_sections_content(report, sections_found):
    # This step is important because some sections are empty given that the next section
    # starts right away at the end of the other
    #print(sections_found)
    temp_sections_found = sections_found.copy()
    for i in range(len(sections_found.index) - 1):
        sec1_end = temp_sections_found.loc[i, "end_title"]
        sec2_start = temp_sections_found.loc[i + 1, "start_title"]
       
        if sec1_end == sec2_start:
            sections_found.iloc[i, [3, 4]] = [sec1_end, sec1_end]
        else:
            sections_found.iloc[i, [3, 4]] = [sec1_end, sec2_start]
          
    # for the last section is up to the length of the report - 1
    i = len(sections_found.index) - 1
    sec1_end = temp_sections_found.loc[i, "end_title"]
    sections_found.iloc[i, [3, 4]] = [sec1_end, len(report)]

    return sections_found

def span_overlap(span1, span2):
    # check if two spans overlaps
    # -1 if the spans does not overlap
    # 0 if the largest span is the first one
    # 1 the largest span is the second one
    if span1[0] <= span2[1] and span2[0] <= span1[1]:
        span1_length = span1[1] - span1[0]
        span2_length = span2[1] - span2[0]
        if span1_length >= span2_length:
            overlap = 0
        else:
            overlap = 1
    else:
        overlap = -1
    return overlap

def remove_overlapping_sections(sections_found):
    # counter for the original variable
    i = 0
    # counter for the new temporal variable
    j = 0
    # a copy of the data frame is needed since the elements inside are going to be used to assign
    # other elements inside since the elements are copy by reference this produce a chain assigment
    # and slow significantly the performance
    columns = list(sections_found.columns.values)
    temp_sections_found = pd.DataFrame(index=np.arange(0, len(sections_found.index)), columns=columns)
    while i <= len(sections_found.index) - 2:
        # check for overlap of section i with section i+1
        overlap = span_overlap(sections_found.iloc[i, [1, 2]], sections_found.iloc[i + 1, [1, 2]])

        if overlap == 0:
            temp_sections_found.iloc[j] = sections_found.iloc[i]
            # jump one to not evaluate the same section again
            i += 1
            j += 1
        elif overlap == 1:
            temp_sections_found.iloc[j] = sections_found.iloc[i + 1]
            # jump one to not evaluate the same section again
            i += 1
            j += 1
        else:
            temp_sections_found.iloc[j] = sections_found.iloc[i]
            j += 1
        i += 1

    # delete unused rows
    temp_sections_found.iloc[j] = sections_found.iloc[i]
    sections_found = temp_sections_found[:j + 1]
    return sections_found

# find subtext in a text
# return the begin and end of subtext
# returns only the first one
def get_text_span(subtext, text):
    # get the span of a given subtext inside a text
    # if the subtext is not inside both start and end is -1
    start_span = text.find(subtext)
    #print(start_span)

    if start_span == -1:
        end_span = -1
    else:
        end_span = start_span + len(subtext)

    span = [start_span, end_span]

    return span

# Finds the location of the title: start_title - end_title
def find_titles_span_and_sort(columns, report, sections):
    sections_found = pd.DataFrame(index=np.arange(0, len(sections)), columns=columns)
    # counter for the number of sections found. In this way the unused rows will be  discarded
    i = 0
    for section in sections:
        # find the occurrence of the section text inside the report
        section_title_span = get_text_span(section, report)
        # if the section is not found skip and go to the next
        if section_title_span[0] == -1:
            continue
        else:
            sections_found.iloc[i] = [section, section_title_span[0], section_title_span[1], 0, 0]
            i += 1
    # delete unused rows
    sections_found = sections_found[:i]
    # sort by start_title
    sections_found = sections_found.sort_values(by=columns[1])
    return sections_found

def find_all_sections_title_single_report(report, sections):
    # this function find the span of each section and returns a data frame with the following columns
    # section_name, start_title, end_title, start_section, end_section

    ###########
    # STEP 1: Find the sections titles and sort
    columns = ["section_name", "start_title", "end_title", "start_section", "end_section"]
    sections_found = find_titles_span_and_sort(columns, report, sections)

    #########
    # STEP 2. Check if any section title overlaps and delete the shortest title
    sections_found = remove_overlapping_sections(sections_found)
    #print(sections_found)

    #######
    # STEP 3: find the contents of each section.
    sections_found = find_sections_content(report, sections_found)
    #print(sections_found)
    return sections_found

def extract(x, sections, takeLastInstance = False):
    
    # x is the original text
    # sections_found function find_all_sections_title_single_report
    sections_found = find_all_sections_title_single_report(x, sections)
    # extracted function extract_section_contents_single_report

    extracted = extract_section_contents_single_report(x, sections_found, sections_list=sections, new_line=True,
                                               delete_empty_sections=True)
    return sections_found, extracted

def sectionize(report):
    # report is the original text
    if report.find('plan and assessment') != -1:
        #reportExtract = extract(report, sections=['plan and assessment'])
        reportExtract = extract(report, sections=['subjective','objective','plan and assessment'], takeLastInstance=True)
    elif report.find('assessment and plan') != -1:
        #reportExtract = extract(report, sections=['assessment and plan'])
        reportExtract = extract(report, sections=['subjective','objective','assessment and plan'], takeLastInstance=True)
    elif report.find('assessment') != -1:
        # extracts the assessment section and converts it to the last version
        # go to extract function, report is the original text
        reportExtract = extract(report, sections=['subjective','objective','assessment'], takeLastInstance=True)
        #print(reportExtract)
    else:
        reportExtractDF = pd.DataFrame({'section_name': ['NULL'],
                                        'start_title':[-99],
                                        'end_title': [-99],
                                        'start_section': [-99],
                                        'end_section': [-99]})
        reportExtract = [reportExtractDF, report]
    return reportExtract

# Extracting start and end positions of specific sections.
data['subjective_start'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='subjective']
                                                  ['start_section'].values)

data['subjective_end'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='subjective']
                                                  ['end_section'].values)


data['objective_start'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='objective']
                                                  ['start_section'].values)

data['objective_end'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='objective']
                                                  ['end_section'].values)

data['assessment_start'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='assessment']
                                                  ['start_section'].values)

data['assessment_end'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='assessment']
                                                  ['end_section'].values)

data['assessmentplan_start'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='assessment and plan']
                                                  ['start_section'].values)

data['assessmentplan_end'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='assessment and plan']
                                                  ['end_section'].values)

data['planassessment_start'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='plan and assessment']
                                                  ['start_section'].values)

data['planassessment_end'] = data['note_des_lower'].apply(lambda x: sectionize(x)[0]
                                                  [sectionize(x)[0]['section_name']=='plan and assessment']
                                                  ['end_section'].values)

def clean_text(note_text: str):
    """
      This function performs basic cleaning for Stanford clinical text
    """
    if not note_text:
        return None
    note_text = re.sub(r"\s(\?|¿)\s", ' ', note_text)
    note_text = re.sub(r"\n", ' ', note_text)
    note_text = re.sub(r"(\?|\¿)", ' ', note_text)
    note_text = re.sub(u"\xa0", ' ', note_text)
    # remove double spaces
    note_text = re.sub(r"\s{2,}", ' ', note_text)
    
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    note_text = ' '.join([word for word in note_text.split() if word not in stop_words])
   
    #Remove plan
    note_text = re.sub(r"(?i)\w*:", '', note_text) # Remove Section headers
    note_text = re.sub(r"\d{1,2}/\d{2}/\d{4}", '', note_text)
    note_text = re.sub(r"\d{1,2}/\d{2}/\d{2}", '', note_text)


    note_text = re.sub(r"\d", '', note_text) #Remove digits
    note_text = re.sub(r"\[\W*\]", ' ', note_text)
    note_text = re.sub(r"\(\W*\)", ' ', note_text)
    note_text = re.sub(r"[\(\)\[\]]", '', note_text)
    note_text = re.sub(r"\/", " ", note_text)
    note_text = re.sub(r"[^\w\s\.]", '  ', note_text)
    note_text = re.sub(r"\s{2,}", ' ', note_text)

    note_text = re.sub("\s(?=\.)", '', note_text)
    
    # filter out short tokens
    note_text = ' '.join([word for word in note_text.split() if len(word) > 1])
    note_text = ' '.join(word.strip(string.punctuation) for word in note_text.split())
    
    if len(note_text.split())>512:
        note_text = ' '.join([word for word in note_text.split() if len(word) > 2])

    note_text = note_text.lower()
    return note_text

data['extractText'] = data['note_des_lower'].apply(lambda x: sectionize(x)[1])
data["visit_no"] = pd.to_numeric(data["visit_no"])
data[['subjective','objective','assessment']]=pd.DataFrame(data.extractText.values.tolist(), data.index).iloc[:, 0:3]

data['subjective']=data['subjective'].str[0]
data['objective']=data['objective'].str[0]
data['assessment']=data['assessment'].str[0]

data['subjective_start']=data['subjective_start'].str[0]
data['subjective_end']=data['subjective_end'].str[0]
data['objective_start']=data['objective_start'].str[0]
data['objective_end']=data['objective_end'].str[0]
data['assessment_start']=data['assessment_start'].str[0]
data['assessment_end']=data['assessment_end'].str[0]
data['assessmentplan_start']=data['assessmentplan_start'].str[0]
data['assessmentplan_end']=data['assessmentplan_end'].str[0]
data['planassessment_start']=data['planassessment_start'].str[0]
data['planassessment_end']=data['planassessment_end'].str[0]

data['assessment_start']=data['assessment_start'].fillna(data['assessmentplan_start'])
data['assessment_start']=data['assessment_start'].fillna(data['planassessment_start'])
data['assessment_end']=data['assessment_end'].fillna(data['assessmentplan_end'])
data['assessment_end']=data['assessment_end'].fillna(data['planassessment_end'])

# Reordering sections based on their start positions to ensure proper sequence in clinical notes.
for index, row in data.iterrows():
        
        #no section
        if math.isnan(row.subjective_start) and math.isnan(row.objective_start) and math.isnan(row.assessment_start):
            print("no section",index)
            data.subjective[index]= data.note_des_lower[index]
        # 3
        elif math.isnan(row.subjective_start) and math.isnan(row.objective_start):
            print("3",index)
            data.subjective[index], data.assessment[index] = data.assessment[index],data.subjective[index]
        # 2
        elif math.isnan(row.subjective_start) and math.isnan(row.assessment_start):
            print("2",index)
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
        # 23
        elif math.isnan(row.subjective_start) and row.objective_start < row.assessment_start:
            print("23",index)
            data.objective[index], data.assessment[index] = data.assessment[index],data.objective[index]
            data.objective[index], data.subjective[index] = data.subjective[index],data.objective[index]
        # 23
        elif math.isnan(row.subjective_start) and row.assessment_start < row.objective_start:
            print("23",index)
            data.subjective[index], data.assessment[index] = data.assessment[index],data.subjective[index]
        # 13
        elif math.isnan(row.objective_start) and row.subjective_start < row.assessment_start:
            print("13",index)
            data.objective[index], data.assessment[index] = data.assessment[index],data.objective[index]
        # 13
        elif math.isnan(row.objective_start) and row.assessment_start < row.subjective_start:
            print("13",index)
            data.subjective[index], data.assessment[index] = data.assessment[index],data.subjective[index]
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
        # 12
        elif math.isnan(row.assessment_start) and row.objective_start < row.subjective_start:
            print("12",index)
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
        # 321
        elif row.assessment_start < row.objective_start and row.objective_start < row.subjective_start:
            print("321",index)
            data.subjective[index], data.assessment[index] = data.assessment[index],data.subjective[index]
        # 312
        elif row.assessment_start < row.subjective_start and row.subjective_start < row.objective_start:
            print("312",index)
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
            data.objective[index], data.assessment[index] = data.assessment[index],data.objective[index]
            
        # 132
        elif row.subjective_start < row.assessment_start and row.assessment_start < row.objective_start:
            print("132",index)
            data.objective[index], data.assessment[index] = data.assessment[index],data.objective[index]
        # 213
        elif row.objective_start < row.subjective_start and row.subjective_start < row.assessment_start:
            print("213",index)
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
        # 231
        elif row.objective_start < row.assessment_start and row.assessment_start < row.subjective_start:
            print("231",index)
            data.objective[index], data.assessment[index] = data.assessment[index],data.objective[index]
            data.subjective[index], data.objective[index] = data.objective[index],data.subjective[index]
        
        else:
            continue
            
# Applying text cleaning to specific sections.
data['subjective_clean1'] = data['subjective'].apply(lambda x: clean_text(x))
data['objective_clean1'] = data['objective'].apply(lambda x: clean_text(x))
data['assessment_clean1'] = data['assessment'].apply(lambda x: clean_text(x))

# Checks if two intervals overlap.
def between(b1,e1,b2,e2):
    if e1<b2 or e2<b1:
        return 0
    return 1

# Labeling sections
data['subj_label']=0
data['obj_label']=0
data['asses_label']=0
for index, row in df_annot_subset.iterrows():
    for index2, rowdata in data.iterrows():
        if row.file==str(rowdata.file).split('.txt')[0]:
            if math.isnan(rowdata.subjective_start) and math.isnan(rowdata.objective_start) and math.isnan(rowdata.assessment_start) and rowdata.SE_inquiry==1:
                data['subj_label'][index2]=1
            if not math.isnan(rowdata.subjective_start) and between(int(row.begin)-69,int(row.end)-69,int(rowdata.subjective_start),int(rowdata.subjective_end)):
                data['subj_label'][index2]=1  
            if not math.isnan(rowdata.objective_start) and between(int(row.begin)-69,int(row.end)-69,int(rowdata.objective_start),int(rowdata.objective_end)):
                data['obj_label'][index2]=1
            if not math.isnan(rowdata.assessment_start) and between(int(row.begin)-69,int(row.end)-69,int(rowdata.assessment_start),int(rowdata.assessment_end)):
                data['asses_label'][index2]=1
            

data=data.drop(columns=['assessmentplan_start', 'assessmentplan_end','planassessment_start','planassessment_end'])

# Preparing data for model training by consolidating, filtering, and reshaping inputs and labels.
seed=0
x1=data['subjective_clean1']
x2=data['objective_clean1']
x3=data['assessment_clean1']

y1=data['subj_label']
y2=data['obj_label']
y3=data['asses_label']

group1=data['mrn']
group2=data['mrn']
group3=data['mrn']

x=pd.concat([x1,x2,x3],)
x.reset_index(inplace=True, drop=True) 

y=pd.concat([y1,y2,y3])
y.reset_index(inplace=True, drop=True) 

group=pd.concat([group1,group2,group3])
group.reset_index(inplace=True, drop=True) 

group1=np.concatenate([np.arange(579),np.arange(579),np.arange(579)])

index=np.where(x.apply(lambda i: len(i.split())<4))[0]

y=y.drop(index)
y=np.array(y)

x=x.drop(index)
x=np.array(x)

group=group.drop(index)
group=np.array(group)
group1=np.delete(group1, index)
group1=np.array(group1)

mrn_to_exclude = mrn_to_exclude.iloc[:, 0].to_numpy()
index_to_exclude = np.where(np.isin(group, mrn_to_exclude))[0] 
group = np.delete(group, index_to_exclude) 
group = group.reshape(-1, 1)
y = np.delete(y, index_to_exclude)
x = np.delete(x, index_to_exclude)
group1 = np.delete(group1, index_to_exclude)
idn=np.arange(1033)
idn=np.array(idn)

X_array_all= x.values.tolist()

# Initialize tokenizer and model encoder based on the specified model name.
def token_encoder(model_name):
     
    if model_name=="llama_13":
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf")
        encoder = LlamaModel.from_pretrained("decapoda-research/llama-13b-hf")
    
    return (tokenizer, encoder)

# Tokenize and encode text data into feature arrays for model input.
def form_X_array(tokenizer, encoder,model_name):
    X_array=[]
    for i in range(len(X_array_all)):
        X_array_one = tokenizer(X_array_all[i],max_length=512,return_tensors='pt') 
        X_array_one = encoder(**X_array_one)
        X_array.append(torch.sum(X_array_one.last_hidden_state[0,:,:],dim=0).tolist())
    return X_array

X_array_llama_13=form_X_array(token_encoder("llama_13")[0],token_encoder("llama_13")[1],"llama_13")
X_array_llama_13= np.array(X_array_llama_13)

# Define a ANN model for note classification.
def Model(model_shape):
    inputlayer = tf.keras.layers.Input(shape=(model_shape,))
    
    layer = tf.keras.layers.Dropout(do, name='dropout')(inputlayer)
    layer = tf.keras.layers.Dense(dense_node, activation='relu')(layer)
    layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(layer)

    model = tf.keras.Model(inputs=inputlayer, outputs=layer)

    return model

param_grid = {'lr': [0.001],
              'batch_size': [32],
              'epochs': [50],
              'dense_node': [128],
              'do':[0.1],
              'model_type' : ['llama_13']
             }

param_grid = list(ParameterGrid(param_grid))
print(len(param_grid))

METRICS= [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc')
         ]

# Model training, evaluation, and performance tracking.
colnames=['model_type','lr','batch_size','dense_node','seed','random_number','epoch','do','train_auc','val_auc','test_auc','train_auc_pa','val_auc_pa','test_auc_pa','n_train','n_val','n_test','n_train_pa','n_val_pa','n_test_pa','n_case_train','n_case_val','n_case_test','n_case_train_pa','n_case_val_pa','n_case_test_pa']
performance = []
for num in range(len(param_grid)):
    #%% Pull data from lookup sheet
    inputGrid = param_grid[num]
    lr = inputGrid['lr']
    batch_size = inputGrid['batch_size']
    epochs = inputGrid['epochs']
    dense_node = inputGrid['dense_node']
    do = inputGrid['do']
    model_type = inputGrid['model_type']
    inputGridDF = pd.DataFrame.from_dict([inputGrid])
    model_type=inputGrid['model_type']
    if model_type=='llama_13':
        X_array=X_array_llama_13
    
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    for l in range(0,1,1):
        for j in range(22,23,2):
            se=l
            ranu=j
            os.environ["PYTHONHASHSEED"] = str(se)
            def reset_seeds():
                np.random.seed(se)
                python_random.seed(se)
                tf.random.set_seed(se)

            reset_seeds()

            cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=ranu)
            idxs1,idxs2,idxs3,idxs4,idxs5 = cv.split(X_array, y, group)
            
            idn_test= idn[idxs1[1]]
            
            idn_val= idn[idxs2[1]]
            idn_train= idn[~np.in1d(np.arange(idn.size),idxs1[1])&~np.in1d(np.arange(idn.size),idxs2[1])]
            
            y_test= y[idn_test]
            y_val= y[idn_val]
            y_train= y[idn_train]
            
            group1_test= group1[idn_test]
            group1_val= group1[idn_val]
            group1_train= group1[idn_train]
            
            df_pa_tr = pd.DataFrame(dict(id_tr=group1_train, y_pa_tr=y_train))
            y_train_pa=np.array(df_pa_tr.groupby('id_tr')['y_pa_tr'].max())
            
            df_pa_val = pd.DataFrame(dict(id_val=group1_val, y_pa_val=y_val))
            y_val_pa=np.array(df_pa_val.groupby('id_val')['y_pa_val'].max())
           
            df_pa_te = pd.DataFrame(dict(id_te=group1_test, y_pa_te=y_test))
            y_test_pa=np.array(df_pa_te.groupby('id_te')['y_pa_te'].max())
            
            if np.abs(y_test.sum()-y_val.sum())<10:

                X_test= X_array[idn_test]
                X_val= X_array[idn_val]
                X_train= X_array[idn_train]

                group_test= group[idn_test]
                group_val= group[idn_val]
                group_train= group[idn_train]

                model = Model(len(X_array[0]))
                model.compile(loss='binary_crossentropy',optimizer=opt,metrics=METRICS)
                aucx_train=[]
                aucx_test=[]
                aucx_val=[0]

                for i in range(epochs):
                    model.fit([X_train], y_train, epochs=1, batch_size=batch_size, validation_data=([X_val], y_val),verbose=0)
                    y_train_pre = model.predict([X_train])
                    train_auc = roc_auc_score(y_train,y_train_pre)
                    
                    pa_tr = pd.DataFrame(dict(id_tr=group1_train, y_pr_tr=y_train_pre.reshape(len(y_train_pre),)))
                    y_train_pa_pre=np.array(pa_tr.groupby('id_tr')['y_pr_tr'].max())
                    train_pa_auc = roc_auc_score(y_train_pa,y_train_pa_pre)

                    y_val_pre = model.predict([X_val])
                    
                    val_auc = roc_auc_score(y_val,y_val_pre)
                    pa_val = pd.DataFrame(dict(id_val=group1_val, y_pr_val=y_val_pre.reshape(len(y_val_pre),)))
                    y_val_pa_pre=np.array(pa_val.groupby('id_val')['y_pr_val'].max())
                    val_pa_auc = roc_auc_score(y_val_pa,y_val_pa_pre)

                    y_test_pre = model.predict([X_test])
                    
                    test_auc = roc_auc_score(y_test,y_test_pre)
                    pa_te = pd.DataFrame(dict(id_te=group1_test, y_pr_te=y_test_pre.reshape(len(y_test_pre),)))
                    y_test_pa_pre=np.array(pa_te.groupby('id_te')['y_pr_te'].max())
                    
                    new_row={'model_type': model_type,'lr':lr,'batch_size':batch_size,'dense_node':dense_node,'seed':se,
                             'random_number':ranu,'epoch':i,'do':do,'train_auc':train_auc,'val_auc':val_auc,'test_auc':test_auc,
 'train_auc_pa':train_pa_auc,'val_auc_pa':val_pa_auc,'test_auc_pa':test_pa_auc,'n_train':len(y_train),'n_val':len(y_val),'n_test':len(y_test),'n_train_pa':len(y_train_pa),'n_val_pa':len(y_val_pa),'n_test_pa':len(y_test_pa),'n_case_train':y_train.sum(),'n_case_val':y_val.sum(),'n_case_test':y_test.sum(),'n_case_train_pa':y_train_pa.sum(),'n_case_val_pa':y_val_pa.sum(),'n_case_test_pa':y_test_pa.sum()}
                    performance.append(new_row)
            print ('last result for val set is : '+ str(aucx_val[-1]))
            print ('last result for test set is : '+ str(aucx_test[-1]))       

performance=pd.DataFrame(performance, columns=colnames)
performance.to_csv("performance_llama_13.csv", index=False)
