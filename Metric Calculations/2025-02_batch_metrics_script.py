
import jsonlines
import json
import glob as glob
import pandas as pd
from collections import Counter
import numpy as np
import difflib
import os
import csv
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import Levenshtein
from pathlib import Path

try:
    from config.settings import (
        ANNOTATIONS_GLOB,
        get_model_results_map,
        OUTPUT_DIR,
        FAST_RESULTS,
    )
except Exception:
    # Fallbacks if settings is unavailable
    ANNOTATIONS_GLOB = os.getenv("ANNOTATIONS_GLOB", "data/annotations/*.jsonl")
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "results/Feb-2025 Results"))
    FAST_RESULTS = os.getenv("FAST_RESULTS", "false").lower() in {"1", "true", "yes"}
    def get_model_results_map():
        return {}


def loadjsonl_to_df(jsonl_file):
    jsonl_data = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            jsonl_data.append(json.loads(line))
            
        df = pd.DataFrame(jsonl_data)
        return df

#Process reports to remove extra spaces/new lines 
def preprocess_text(text):
    #remove newline characters (\n) and paragraph markers (\n\n or /)
    text = re.sub(r'\n', ' ', text) #remove newline characters and replace with a single space, with the exception of dates
    return text 

def apply_preprocessing(row):
    return preprocess_text(row['text'])

#Redact text using doccano output
def redact_text(row):
    text = preprocess_text(row['text']) #extract text from each row, using the processed text
    manual_output = row['manual_output'] #extract manual_output for each row
    redacted_text = text #initialise redacted text as the original text
    offset = 0 #initialise offset to keep track fo cumulative position changes from preprocessing 

    #iterate over each sublist in manual_output
    for annotation in manual_output:
        start_token, end_token, string = annotation #extract the start, end tokens and string for each sublist
        if string != 'time':
            update_start = start_token + offset
            update_end = end_token + offset
            redacted_text = redacted_text[:update_start] + '[' + string + ']' + redacted_text[update_end:] #replaced the text from start-end token with string and square brackets
            len_diff = len(string) - (end_token - start_token)
            offset += len_diff + 2 #calculate and add offset, and add +2 to account for the addition of the square brackets

    redacted_text = re.sub(r'(?<=\s)\d{4}(?=\s|$|[.,!?();:])', '[date]', redacted_text) # replace 4-digit numbers preceded by a space with '[date]'

    return redacted_text

#Combine dataframes
#df1 and df2 refer to the df to be merged
#merge_criteria is the column to merge the df on
#drop columns before/after merge is a list of columns to remove if required (put in [])

def combine_df(df1, df2, merge_criteria, drop_col_before_merge=None, drop_col_after_merge=None):
    
    if drop_col_before_merge:
        df1 = df1.drop(columns=drop_col_before_merge, errors='ignore')
        df2 = df2.drop(columns=drop_col_before_merge, errors='ignore')
    
    df_comb = pd.merge(df1, df2, on=merge_criteria)
    
    if drop_col_after_merge:
        df_comb = df_comb.drop(columns=drop_col_after_merge, errors='ignore')
    
    return df_comb

#extract lists of redacted words
def extract_redacted_words(row, column):
    text_words = row['text'].split()
    column_words = row[column].split()
    redacted_words = [word for word in text_words if word not in column_words]
    return redacted_words


#Find the first shared word and update 'model' column
def update_model(model_text, text_text):
    model_words = model_text.split() #split text into lists of words
    text_words = text_text.split()
    shared_word = next((word for word in model_words if word in text_words), None) #find first shared word between model/text

    if shared_word:
        index = model_text.find(shared_word) #find index of shared word in model text
        return model_text[index:].strip() #keep substring of model text starting from the shared word
    return model_text #if no shared word is found, return the original model text

def calculate_metrics(model_redaction, manual_redaction):
    tp = fp = fn = 0 #initialise tp, fp, fn counts as 0

    tp_words = []
    fp_words = []
    fn_words =[]

    #copy each of model_redaction and manual_redaction for modification 
    model_redaction_processed = model_redaction.copy()
    manual_redaction_processed = manual_redaction.copy()

    #calculate tp
    for word in manual_redaction: #for each word in manual_redaction
        while word in model_redaction_processed: #if the word is also in manual_redaction
            tp += 1 #increment one to tp
            tp_words.append(word)

            #remove word from both lists to prevent double counting
            model_redaction_processed.remove(word) #modify the copies only
            manual_redaction_processed.remove(word)

    #calculate fn
    for word in manual_redaction_processed:
        if word not in model_redaction_processed:
            fn += 1 #increment one to fn
            fn_words.append(word)
    
    #calculate fp
    for word in model_redaction_processed:
        if word not in manual_redaction:
            fp += 1 #increment one to fp
            fp_words.append(word)

    return tp, fp, fn, tp_words, fp_words, fn_words

def output_metrics(row):
    return calculate_metrics(row['model_redaction'], row['manual_redaction'])

# Function to calculate metrics by row
def calculate_metrics_by_row(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def report_metrics_with_CIs (row, comb, by_dataset=False):
    n_bootstrap = 10000
    confidence_level = 95
    alpha = (100 - confidence_level) / 2

    # Disable iteration through each row
    #for index, row in grouped_comb.iterrows():
    model_name = row['model_name'][0]
    num_shots = row['num_shots'][0]
    tp_count = row['tp'][0]
    fp_count = row['fp'][0]
    fn_count = row['fn'][0]
    
    # Calculate observed precision, recall, and F1
    precision, recall, f1 = calculate_metrics_by_row(tp_count, fp_count, fn_count)
    
    # Bootstrap lists
    precisions = []
    recalls = []
    f1_scores = []
    bleu_scores_model_ref = []
    
    # Bootstrap sampling
    for _ in range(n_bootstrap):
        # Sample within the current group only
        sample = comb[(comb['model_name'] == model_name) & (comb['num_shots'] == num_shots)].sample(frac=1, replace=True)
        sample_tp = sample['tp'].sum()
        sample_fp = sample['fp'].sum()
        sample_fn = sample['fn'].sum()
        
        # Call function to calculate metrics
        sample_precision, sample_recall, sample_f1 = calculate_metrics_by_row(sample_tp, sample_fp, sample_fn)
        
        precisions.append(sample_precision)
        recalls.append(sample_recall)
        f1_scores.append(sample_f1)
    
    # Calculate confidence intervals
    precision_ci = np.percentile(precisions, [alpha, 100 - alpha])
    recall_ci = np.percentile(recalls, [alpha, 100 - alpha])
    f1_ci = np.percentile(f1_scores, [alpha, 100 - alpha])
    
    # Print results, including if by dataset
    if by_dataset:
        print(f"Results for model '{model_name}' with num_shots '{num_shots}' on dataset '{row['dataset'][0]}':")
    else:
        print(f"Results for model '{model_name}' with num_shots '{num_shots}':")

    # print(f"TP count: {tp_count}")
    # print(f"FP count: {fp_count}")
    # print(f"FN count: {fn_count}")
    print(f"Precision: {precision:.3f} (95% CI: {precision_ci[0]:.3f}, {precision_ci[1]:.3f})")
    precision = (f"{precision:.3f} (95% CI: {precision_ci[0]:.3f}, {precision_ci[1]:.3f})")
    print(f"Recall: {recall:.3f} (95% CI: {recall_ci[0]:.3f}, {recall_ci[1]:.3f})")
    recall = (f"{recall:.3f} (95% CI: {recall_ci[0]:.3f}, {recall_ci[1]:.3f})")
    print(f"F1-score: {f1:.3f} (95% CI: {f1_ci[0]:.3f}, {f1_ci[1]:.3f})")
    f1= (f"{f1:.3f} (95% CI: {f1_ci[0]:.3f}, {f1_ci[1]:.3f})")
    print()  # Add an empty line for spacing
    
    return precision, recall, f1


def remove_excluded_words(column):
    #List of professional labels
    not_applicable= ['consultant:', 'consultant', '(consultant', '"consultant,"', 'consultant.', 'consultant)', '(consultant)', 'report/consultant', 'cons', '"consultant,"', "consultant,", 'consultant:',
    'pathologist:', 'pathologist', 'dermatopathologists.', 'neuropathologist', '"pathologist,"', 'haematopathologist', 'pathologist)', 'pathologists', 'haematopathologist.', '"pathologist,"', '"pathologist,"',
    'bms.', 'bms', '(bms).', 'bms.dictated', 'bms.one', 'bms.-------------------x--------------------', '(sonographer)',
    'pathology', '"pathology,"', 'neuropath',
    'cellular', '(cellular', 
    'msk', '(msk', 'reporting', 'musculoskeletal',
    'registrar:', '"registrar"', 'registrar)', 'registrar', 'registrar.', 'spr', 'fellow.', 'fellow)', 'fellow', 'fellow).', '"fellow,"', '"registrar,"', '(fellow)', '(fellow)',
    'radiologist', 'radiologist)', '(radiologist)', 'radiologist.', '"neuroradiologist"', '"radiologist"', '(radiologist', 'neuroradiologist', 'neuroradiologist)', 'radiologists', 'radiologist:', 'neuroradiologist.', '"radiologist),"', 'radiologist;',
    'prof', 'professor', 'raioogist', 'orthopod',
    'gmc', 'gmc:', 'ltd', 'nan', 'ref:', 'resident', 'as', 'of', 'senior', 
    'clinicians', 'clinician' 'clinician:', '(clinical', 'clinic', 'clinic.', 'hand', '(ss',
    'team.',
    'speciality', 'specialty', 'specialist', '(specialist', '"neuropath,"',
    'al', '(general', '"radiologist),"', 'suggestive', 'supervised', 'surg', 'pathol',
    'scientist', 'scientist.', '"scientist"""', '"registrar,"', 'registered', 'dated:',
    'received', '"neuroradiologist,"', 'sonographer)', 'physio', '"pathologist,"', '(dermatopathology)', 'agreed',
    'senior', '15:16', 'regi?tered', 'new', 'clinic',
    'number:', 'st4', 'st2', 'st5', 'st3', 'st1', '(haematopathologist)', 'urological',
    'principal', '"pathology,"', '"sonographer),"', 'reported', 'had', 'cross-sectional', 'cxr', 'consensus',
    '(trainee', 'trainee)', 'trainee', 'lead', 'mdt', 'lead)',
    'for', 'inform', 'formal', '"radiologist),"', '"registrar,"', 'anaesthetist', 'classification.',
    'is', '-', 'at', 'pre-registration', 'onwards', 'miu', ')', '(senior',
    'swifft', 'post', 'cct', 'radiograph', 'year', '"neuroradiologist,"', '"radiologist,"', 'feels', '"radiologist),"', "(radiographer)",
    'surgeon', 'radiology', '(radiology', 'omfs', 'ortho', 'sho', 'doctor,', 'shows', '"radiologist),"', 'radiology.', 'a&e', 'gynaecologist', '"neuroradiologist,"', 'neurology', '"radiologist,"', 'trauma', 'likely',
    'radiographer', 'neurointerventional', '"neurologist,"', '"spr,"', 'gp', 'neurosurgery', 'sonographer', 'radiographers', '(sonographer).', '"neuroradiologist,"', '"sonographer),"', 'radiographer.']

    return (column.apply(lambda word_list: [word for word in word_list if word.lower() not in not_applicable]))

#Remove professional titles for models that don't do this
def exclude_professional_details(column):
    professional_titles=['dr', 'dr.', 'prof', 'professor', 'prof.']
    
    return (column.apply(lambda word_list: [word for word in word_list if word.lower() not in professional_titles]))


def tokenize(text):
    return text.split()


def calculate_bleu(reference_texts, candidate_texts):
    return corpus_bleu([[tokenize(ref)] for ref in reference_texts], [tokenize(candidate) for candidate in candidate_texts])


def bootstrap_bleu(reference, hypothesis, num_samples=100, ci=95):
    bleu_scores = []
    n = len(reference)
    for _ in range(num_samples):
        indices = np.random.choice(range(n), size=n, replace=True)
        resampled_reference = [reference[i] for i in indices]
        resampled_hypothesis = [hypothesis[i] for i in indices]
        bleu_score = corpus_bleu([[tokenize(ref)] for ref in resampled_reference], 
                                 [tokenize(hyp) for hyp in resampled_hypothesis])
        bleu_scores.append(bleu_score)
    lower_bound = np.percentile(bleu_scores, (100 - ci) / 2)
    upper_bound = np.percentile(bleu_scores, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


def safe_levenshtein(str1, str2):
    if pd.isna(str1) or pd.isna(str2):
        return float('nan')
    return Levenshtein.distance(str1, str2)


def bootstrap_ci_levenshtein(data, num_samples=100, ci=95):
    bootstrap_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data.dropna(), size=len(data.dropna()), replace=True)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound


file_list = glob.glob(ANNOTATIONS_GLOB)

#initialise empty list to store all doccano df
dfs = []

#iterate over each doccano file and append to the combined df
for file in file_list:
    df = pd.read_json(file, lines=True)

    #add new column called 'dataset' that contains the dataset name for reference
    filename = os.path.basename(file)  
    dataset = os.path.splitext(filename)[0]
    df['dataset'] = dataset
    dfs.append(df)

#concatenate all df into one
full_ann1 = pd.concat(dfs, ignore_index=True)
full_ann = full_ann1.drop(['Comments', 'study_id'], axis=1)
full_ann.rename(columns={'label': 'manual_output'}, inplace=True)


#Reverse engineer doccano into text, creating a new column called redacted_text
full_ann['redacted_text'] = full_ann.apply(redact_text, axis=1)
#full_ann.head(1300) #comment in or out to examine data
#full_ann['redacted_text'].to_csv('full_ann.csv', index=False) #This exports the redacted_text column to a .csv file for sense checking.

# %%
# Define the overall analysis
model_results_map = get_model_results_map()
models_for_evaluation = pd.DataFrame(columns=['model_name', 'prof_details', 'path_to_results_file'], data=[
        ["Azure Health DeID v1","with_professional_details", model_results_map.get("Azure Health DeID v1", "")],
        ["anoncat","with_professional_details", model_results_map.get("anoncat", "")],
        ["anoncat","without_professional_details", model_results_map.get("anoncat", "")],
        ["11 Feb 2025 OUH Fine Tuned AnonCAT 0.00002 10ep new concepts 10per","with_professional_details", model_results_map.get("11 Feb 2025 OUH Fine Tuned AnonCAT 0.00002 10ep new concepts 10per", "")],
        ["11 Feb 2025 OUH Fine Tuned AnonCAT 0.00002 10ep new concepts 10per","without_professional_details", model_results_map.get("11 Feb 2025 OUH Fine Tuned AnonCAT 0.00002 10ep new concepts 10per", "")],
        ["obi/deid_roberta_i2b2","with_professional_details", model_results_map.get("obi/deid_roberta_i2b2", "")],
        ["obi/deid_roberta_i2b2","without_professional_details", model_results_map.get("obi/deid_roberta_i2b2", "")],
        ["obi/deid_bert_i2b2","with_professional_details", model_results_map.get("obi/deid_bert_i2b2", "")],
        ["obi/deid_bert_i2b2","without_professional_details", model_results_map.get("obi/deid_bert_i2b2", "")],
    ])

#Make a list of unique results files for results, and pull out a list of unique files
results_files = models_for_evaluation['path_to_results_file'].unique()

#Make df to hold results
result = pd.DataFrame()

#Cycle through each unique file and load it in to the combined df
for file in results_files:
    new_file = pd.read_csv(file)
    result = pd.concat([result, new_file], ignore_index = True)

model_results = result.drop(columns=['Unnamed: 0', 'task_name', 'model_kwargs'])
model_results.rename(columns={'id_number_in_dataset': 'id'}, inplace=True)

#Combine datasets (doccano output and results)
#Rename columns so that there are four: dataset (where data from), id (unique identifier), model (model output) and manual (manual annotation)
comb1 = combine_df(model_results, full_ann, merge_criteria='id')
#comb = comb1.drop(columns=['original_report', 'manual_output', 'dataset_y'])
comb = comb1.copy()
comb.rename(columns={'redacted_text': 'manual', 'model_output': 'model', 'dataset_x': 'dataset'}, inplace=True)

#Convert columns to string
comb['model'] = comb['model'].astype(str)
comb['text'] = comb['text'].astype(str)
comb['manual'] = comb['manual'].astype(str)

#pre-process to remove any preamble
comb['model'] = comb.apply(lambda row: update_model(row['model'], row['text']), axis=1)

comb['model'] = comb['model'].apply(preprocess_text)
comb['text'] = comb['text'].apply(preprocess_text)
comb[['model', 'text', 'manual']] = comb[['model', 'text', 'manual']].applymap(lambda x: x.lower())

#extract lists of redacted words in model and manual labelling
comb['model_redaction'] = comb.apply(lambda row: extract_redacted_words(row, 'model'), axis=1)
comb['manual_redaction'] = comb.apply(lambda row: extract_redacted_words(row, 'manual'), axis=1)

#comb.head(1) #comment in/out to view df
#comb[['model', 'manual']].to_csv('comb.csv', index=False) #Export for sense checking


#Define the categories are groups of labels
all_labels = set([label for row in comb['manual_output'] for (_, _, label) in row])

categories = {
    "Names":['patient name', 'hcp name'],
    "Other Unique Identifier":["Hospital/Unit", "External healthcare organisation", "Profession"],
    "Dates":['Date', 'Age over 89'],
    "Medical Record Numbers":['mrn', 'NHS number', 'specimen-identifier'],
    "Phone":['Phone'], 
}

#For fast results, skip Bleu calculation.
fast_results = FAST_RESULTS

#Selective redaction based on groups of labels (labels below is a list of labels)
def redact_text_by_labels(row, labels):
    text = preprocess_text(row['text']) #extract text from each row, using the processed text
    manual_output = row['manual_output'] #extract manual_output for each row
    redacted_text = text #initialise redacted text as the original text
    offset = 0 #initialise offset to keep track fo cumulative position changes from preprocessing 

    #iterate over each sublist in manual_output
    for annotation in manual_output:
        start_token, end_token, string = annotation #extract the start, end tokens and string for each sublist

        if string in labels:
            update_start = start_token + offset
            update_end = end_token + offset
            redacted_text = redacted_text[:update_start] + '[' + string + ']' + redacted_text[update_end:] #replaced the text from start-end token with string and square brackets
            len_diff = len(string) - (end_token - start_token)
            offset += len_diff + 2 #calculate and add offset, and add +2 to account for the addition of the square brackets

    return redacted_text

#Define results dfs
# First a df for overall results
overall_results = pd.DataFrame(columns=['model_name', 'prof_details', 'num_shots', 'precision', 'recall', 'f1', "bleu", "levenshtein"], data=[])

# Next a df for recall by category
by_category_results = pd.DataFrame(columns=['model_name', 'prof_details', 'num_shots', 'Names', 'Other Unique Identifier', 'Dates', "Medical Record Numbers", "Phone"], data=[])

#Next a df for by-dataset results
dataset_results = pd.DataFrame(columns=['model_name', 'prof_details', 'num_shots', 'dataset', 'precision', 'recall', 'f1'], data=[])

#Store unique fp and fn words in a dict
model_fp_words = {}
model_fn_words = {}

# #Remove non-applicable words as specified in Rachel's v1 code
comb['manual_redaction'] = remove_excluded_words(comb['manual_redaction'])
comb['model_redaction'] = remove_excluded_words(comb['model_redaction'])

#Cycle through each eval config
for index, eval in models_for_evaluation.iterrows():
    #For that eval, select only the relevant model
    comb_copy = comb[comb['model_name']==eval['model_name']].copy()

    #Iterate through each num_shots value
    for num_shots in comb_copy['num_shots'].unique():

        #Filter out just the results for that number of shots
        comb_copy = comb_copy[comb_copy['num_shots']==num_shots]

        #Calculate the Leevenshtein distance first, prior to any editing
        comb_copy['distance_text_model'] = comb_copy.apply(lambda row: safe_levenshtein(row['text'], row['model']), axis=1)
        mean_ld=comb_copy['distance_text_model'].mean()
        ld_lower_bounds, ld_upper_bounds = bootstrap_ci_levenshtein(comb_copy['distance_text_model'], num_samples=1000, ci=95)
        ld_with_ci= (f"{mean_ld:.3f} (95% CI: {ld_lower_bounds:.3f}, {ld_upper_bounds:.3f})")
        #print (ld_with_ci)

        #Next, if professional details are being excluded, filter these out
        if eval['prof_details'] == "without_professional_details":

            # #Remove professional titles
            comb_copy['manual_redaction'] = exclude_professional_details(comb_copy['manual_redaction'])
            comb_copy['model_redaction'] = exclude_professional_details(comb_copy['model_redaction'])
        
        # #Now calculate the metrics
        comb_copy[['tp','fp', 'fn', 'tp_words', 'fp_words', 'fn_words']] = comb_copy.apply(output_metrics, axis=1, result_type='expand')

        # #Calculate the overall model results
        grouped_comb_overall = comb_copy.groupby(['model_name','num_shots']).agg({'tp': 'sum', 'fp': 'sum', 'fn': 'sum'}).reset_index()
        print (grouped_comb_overall.to_dict())

        #Save FP and FN words
        model_fp_words[eval['model_name']] = set(word for sublist in comb_copy.fp_words.values for word in sublist)
        model_fn_words[eval['model_name']] = set(word for sublist in comb_copy.fn_words.values for word in sublist)

        #Calculate precision, recall, f1
        precision, recall, f1 = report_metrics_with_CIs(grouped_comb_overall.to_dict(), comb_copy)

        #If producing results quickly, skip over BLEU score.
        if not fast_results:
            #Now calculate bleu score with CIs
            reference = comb_copy['text'].tolist()
            model = comb_copy['model'].tolist()
            bleu_score_model_ref = corpus_bleu([[tokenize(ref)] for ref in reference], [tokenize(txt) for txt in model])
            bleu_ci_model_ref_lower, bleu_ci_model_ref_upper = bootstrap_bleu(reference, model)
            bleu_score_with_ci= (f"{bleu_score_model_ref:.3f} (95% CI: {bleu_ci_model_ref_lower:.3f}, {bleu_ci_model_ref_upper:.3f})")
            print (bleu_score_with_ci)
        else:
            bleu_score_with_ci = "*"

        #Write results for the overall model
        overall_results.loc[len(overall_results)] = [eval['model_name'], eval['prof_details'], num_shots, precision, recall, f1, bleu_score_with_ci, ld_with_ci]
    
        #A dict to hold recalls by category results
        by_category_results_dict={
            'model_name': eval['model_name'],
            'prof_details': eval['prof_details'],
            'num_shots': num_shots
        }

        #Skip dataset results if doing fast_results
        if not fast_results:

            #Now calculate recall by category, where a category is a group iof labels
            #Cycle through labels
            for category, labels in categories.items():
                label_df=comb_copy.copy()

                print (category)

                #Redact for just the one item in the reference text, and make lower case
                label_df['manual']=label_df.apply(redact_text_by_labels, labels=labels, axis=1).str.lower() 
                label_df['manual'] = label_df['manual'].astype(str) 

                #extract lists of redacted words in manual labelling, with the single label. Because we are only checking one category, we cannot interpret the FP result here. 
                label_df['manual_redaction'] = label_df.apply(lambda row: extract_redacted_words(row, 'manual'), axis=1)
                
                #Now calculate tp, fp, etc. Make sure it is calculated only for that one label
                label_df[['tp','fp', 'fn', 'tp_words', 'fp_words', 'fn_words']] = label_df.apply(output_metrics, axis=1, result_type='expand')

                # #Calculate the overall model results; note, only the True positives are valid here.
                grouped_comb_overall = label_df.groupby(['model_name','num_shots']).agg({'tp': 'sum', 'fp': 'sum', 'fn': 'sum'}).reset_index()
                precision, recall, f1 = report_metrics_with_CIs(grouped_comb_overall.to_dict(), label_df)
                print ("tp "+ str(label_df['tp'].sum()) + "  fn:" + str(label_df['fn'].sum()))

                #Store recall in a dict for each category
                by_category_results_dict[category] = recall

            #Write results for the by category model
            by_category_results.loc[len(by_category_results)] = by_category_results_dict

            #Next, calculate by-category results and add to a by_categories_df
            #Iterate through each category
            for dataset in comb_copy['dataset'].unique():
                #Filter out just the results for that dataset
                comb_dataset = comb_copy[comb_copy['dataset']==dataset]

                #Calculate the dataset level results (NB: the below will return only 1 dataset as we are filtering by dataset prior to this)
                grouped_comb_by_category = comb_dataset.groupby(['model_name', 'dataset', 'num_shots']).agg({'tp': 'sum', 'fp': 'sum', 'fn': 'sum'}).reset_index()

                #Calculate the metric (NB)
                precision, recall, f1 =  report_metrics_with_CIs(grouped_comb_by_category.to_dict(), comb_dataset, by_dataset=True)
                dataset_results.loc[len(dataset_results)] = [eval['model_name'], eval['prof_details'], num_shots, dataset, precision, recall, f1]


# #Save outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
overall_results.to_csv(str(Path(OUTPUT_DIR) / "2025-02-11 overall_results.csv"), index=False)
by_category_results.to_csv(str(Path(OUTPUT_DIR) / "2025-02-11 by_category_results.csv"), index=False)
dataset_results.to_csv(str(Path(OUTPUT_DIR) / "2025-02-11 dataset_results.csv"), index=False)