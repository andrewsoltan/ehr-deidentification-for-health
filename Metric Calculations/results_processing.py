#Function to load jsonl file into a pandas dataframe
#Syntax will be chosen_name = loadjsonl_to_df(jsonl_file)
import glob, json
import pandas as pd

def loadjsonl_to_df(jsonl_file):
    jsonl_data = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            jsonl_data.append(json.loads(line))
            
        df = pd.DataFrame(jsonl_data)
        return df
    

#Function to load all jsonl files into a single pandas dataframe
#Syntax will be chosen_name = loadjsonl_to_df(jsonl_file)

def loadall_jsonl_to_df(file_pattern):
    jsonl_data = []
    
    for jsonl_file in glob.glob(file_pattern):
        with open(jsonl_file, 'r') as file:
            for line in file:
                jsonl_data.append(json.loads(line))
    df = pd.DataFrame(jsonl_data)
    return df


#Function for statistical description of text
#e.g., describe_text(rk, 'text')

def describe_text(df, text):
    token_counts = df[text].apply(lambda x: len(str(x).split()))
    token_total = token_counts.sum()
    token_mean = token_counts.mean()
    max_token = token_counts.max()
    min_token = token_counts.min()
    
    token_stats = pd.DataFrame({
        'Statistic': ['Total token count', 'Mean tokens', 'Max tokens', 'Min tokens'],
        'Value': [token_total, token_mean, max_token, min_token]
    })

    print(f'Table of descriptive token statistics')
    return(token_stats)


#Function for % of reports with at least one identifier
#e.g., percent_with_phi(rk, 'label')

def percent_reports_with_phi(df, label):
    #Count how many examples have at least one example of PHI
    counts = df[label].apply(lambda x: 'No PHI' if not x else 'PHI').value_counts()
    
    if 'PHI' in counts:
        phi_count = counts['PHI']
        total_count = counts.sum()
        percent_phi = (phi_count/total_count)*100
    else: 
        percent_phi = 0
        
    return(f'{percent_phi.round(2)}% of reports contain at least one instance of PHI')


#Function for proportion of PHI

def phi_types(df, text, label):
    
    #Count total tokens
    token_counts = df[text].apply(lambda x: len(str(x).split()))
    token_total=token_counts.sum()

    #Explode the label column to separate out the start, end tokens and PHI categories
    exploded_df = df.explode(label)
    
    #Remove any rows where there are no contents
    exploded_df = exploded_df[exploded_df[label].notnull()]

    #Make a new df and describe how many PHI and categories
    exploded_df[['start_token', 'end_token', 'label_type']] = pd.DataFrame(exploded_df[label].tolist(), index=exploded_df.index)
    total_phi = exploded_df.shape[0] #Total count of PHI in the datasheet
    percent_phi = (total_phi / token_total) * 100

    #Count PHI instances by label
    phi_numbers = exploded_df['label_type'].value_counts()
    phi_percent = (phi_numbers / total_phi) * 100

    #Make new DF for PHI counts and %
    phi_data = pd.DataFrame({
        'PHI Category': phi_numbers.index, 
        'Count': phi_numbers.values,
        '% total': phi_percent.values
    })

    print(f'The total number of PHI instances in the dataset is {total_phi}, or {percent_phi.round(2)}% of the total word count.')
    return(phi_data)

#Nest all functions
def run_all(df, text, label):
    desc = describe_text(df, text)
    percent = percent_reports_with_phi(df, label)
    phi = phi_types(df, text, label)
    return desc, percent, phi