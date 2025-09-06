import os, tabulate, torch
from openai import AzureOpenAI
import pandas as pd
import bitsandbytes, accelerate
from typing import Union
from evaluate import load
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch, gc
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate import load
from sacrebleu.metrics import BLEU
import evaluate
import requests, json
from config import settings
import re

def generate_prompt_threads (instruction: str, in_context_examples: list, item_for_task: str, number_shots: list, use_system_prompt: bool=False ) -> list:
    all_prompts = []
    for num_shots in number_shots:
        chat_thread = []
        if (num_shots > len(in_context_examples)):
            raise ValueError("This are not enough examples in the list-of-tuples passed to return sufficient examples")
        if (use_system_prompt):
            chat_thread.append({"role":"system", "content": instruction})
            if num_shots > 0:
                for i in range(num_shots):
                    chat_thread.append({"role":"user", "content": in_context_examples[i][0]})
                    chat_thread.append({"role":"assistant", "content": in_context_examples[i] [1]})
            chat_thread.append({"role":"user", "content":item_for_task})
        elif (~use_system_prompt):
            if num_shots == 0:
                task_prompt = instruction + " \n " + item_for_task
                chat_thread.append({"role":"user", "content":task_prompt})
            elif num_shots > 0:
                for i in range(num_shots):
                    prompt = instruction + " \n " + in_context_examples[i][0]
                    response = in_context_examples[i][1]
                    chat_thread.append({"role":"user", "content": prompt})
                    chat_thread.append({"role":"assistant", "content": response})
                task_prompt = instruction + " \n " + item_for_task
                chat_thread.append({"role":"user", "content":task_prompt})
        all_prompts.append(chat_thread)
    return all_prompts


def hf_load_model (name: str, model_kwargs: dict, multi_gpu: bool=True, default_gpu_id: int = 0, HFapiToken: str=settings.HF_ACCESS_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if (~multi_gpu):
        pipe = pipeline(
            "text-generation",
            model=name,
            token=HFapiToken,
            tokenizer=tokenizer,
            device= default_gpu_id,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
    elif (multi_gpu):
        pipe = pipeline(
            "text-generation",
            model=name,
            token=HFapiToken,
            tokenizer=tokenizer,
            device_map= "auto",
            model_kwargs=model_kwargs,
            trust_remote_code=True, 
        )
    return pipe


def hf_load_bert_model (name: str, model_kwargs: dict, default_gpu_id: int = 0, HFapiToken: str=settings.HF_ACCESS_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipe = pipeline(
        "token-classification",
        model=name,
        token=HFapiToken,
        tokenizer=tokenizer,
        device= default_gpu_id,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
    )
    return pipe


def hf_generate_prompts (pipe, prompt_dict_list: Union[list,dict]):
    formatted_prompt_list = []
    for prompt_dict in prompt_dict_list:
        prompt = pipe.tokenizer.apply_chat_template(
            prompt_dict, tokenize=False, add_generation_prompt=True
        )
        formatted_prompt_list.append(prompt)
    return formatted_prompt_list


def hf_run_model_using_prompts (pipe, prompt_list: list, max_new_tokens: int=200, temperature: int=0.1, top_k:int =50, top_p:int =0.95) -> list:
    model_outputs=[]
    for prompt in prompt_list:
        outputs = pipe(
            prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p, pad_token_id=pipe.tokenizer.eos_token_id
        )
        output_text = outputs[0]["generated_text"][len(prompt) :]
        model_outputs.append(output_text)
    return(model_outputs)


def openai_load_client_and_perform_inference (openAI_key: str = settings.AZURE_OPENAI_API_KEY,  openAI_endpoint: str = settings.AZURE_OPENAI_ENDPOINT, prompt_dict_list: Union[list,dict] = None, api_version: str = settings.AZURE_OPENAI_API_VERSION, deployment_name: str = settings.AZURE_OPENAI_DEPLOYMENT, max_new_tokens: int=200, temperature: int =0.1, top_k: int =50, top_p: int =0.95):
    model_outputs = []
    for prompt_dict in prompt_dict_list:
        try:
            client = AzureOpenAI(
                api_key=openAI_key,
                api_version=api_version,
                azure_endpoint=openAI_endpoint
            )
            client.max_response = max_new_tokens
            completion = client.chat.completions.create(
                messages=prompt_dict,
                model=deployment_name,
                temperature=temperature,
                top_p=top_p,
            )
            response = completion.choices[0].message.content
            model_outputs.append (response)
        except Exception as e:
            model_outputs.append ("Error " + str(e))
    return (model_outputs)


def evaluate_output (model_output: str, reference_text: str, metrics: list) -> dict:
    calculated_metrics={}
    if 'rouge_l' in metrics:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text,model_output)
        calculated_metrics['rouge_l'] = scores
    if 'bleu' in metrics:
        bleu = evaluate.load('bleu')
        try:
            bleu_score = bleu.compute(predictions=[model_output], references=[reference_text])
            calculated_metrics['bleu'] = bleu_score['bleu']
        except:
            calculated_metrics['bleu'] = 0   
    if 'BERTscore' in metrics:
        bert_score = evaluate.load("bertscore")
        calculated_bertscore = bert_score.compute(predictions=[model_output], references=[reference_text], lang="en")
        calculated_metrics['BERTscore'] = calculated_bertscore
    return calculated_metrics


def make_request_to_microsoft_deid (URL, target, text, bearer_token=settings.AZURE_HDS_BEARER_TOKEN):
    headers = {
        "Authorization": "Bearer " + bearer_token,
        "Content-Type" : "application/json"
    }
    payload = {
            "DataType":"Plaintext",
            "Operation":"Redact",
            "InputText": text
    }
    try:
        anonreturn = requests.post(URL+target, headers=headers, json=payload).json()  
        output=anonreturn['outputText']  
    except:
        output = "Error, is your bearer token valid?"
    return (output)


def reconstruct_anoncat_text(original_text, new_text, start_index, end_index):
    new_text = "["+new_text+"]"
    return original_text[:start_index] + new_text + original_text[end_index:]


def inference_with_bert(pipe, passage, chunk_size=400):
    tokenizer = pipe.tokenizer
    encodings = tokenizer(passage, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encodings["input_ids"]
    offsets = encodings["offset_mapping"]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_offsets = offsets[i:i+chunk_size]
        if not chunk_offsets:
            continue
        start = chunk_offsets[0][0]
        end = chunk_offsets[-1][1]
        chunk_text = passage[start:end]
        chunks.append(chunk_text)
    redacted_chunks = []
    for chunk in chunks:
        entities = pipe(chunk)
        current_entity = None
        redacted_spans = []
        for entity in entities:
            tag = entity['entity']
            start = entity['start']
            end = entity['end']
            label = tag.split('-', 1)[-1] if '-' in tag else None
            if tag.startswith('B-'):
                if current_entity:
                    redacted_spans.append(current_entity)
                current_entity = {'start': start, 'end': end, 'label': label}
            elif tag.startswith('I-'):
                if current_entity and current_entity['label'] == label:
                    current_entity['end'] = end
                else:
                    if current_entity:
                        redacted_spans.append(current_entity)
                    current_entity = {'start': start, 'end': end, 'label': label}
            elif tag.startswith('L-'):
                if current_entity and current_entity['label'] == label:
                    current_entity['end'] = end
                    redacted_spans.append(current_entity)
                    current_entity = None
                else:
                    if current_entity:
                        redacted_spans.append(current_entity)
                    redacted_spans.append({'start': start, 'end': end, 'label': label})
                    current_entity = None
            elif tag.startswith('U-'):
                if current_entity:
                    redacted_spans.append(current_entity)
                redacted_spans.append({'start': start, 'end': end, 'label': label})
                current_entity = None
            elif tag == 'O':
                if current_entity:
                    redacted_spans.append(current_entity)
                    current_entity = None
        if current_entity:
            redacted_spans.append(current_entity)
        chunk_length = len(chunk)
        for span in redacted_spans:
            span['end'] = min(span['end'], chunk_length)
        redacted_spans.sort(key=lambda x: x['start'])
        prev_end = 0
        redacted_parts = []
        for span in redacted_spans:
            redacted_parts.append(chunk[prev_end:span['start']])
            redacted_parts.append(f"[{span['label']}]")
            prev_end = span['end']
        redacted_parts.append(chunk[prev_end:])
        redacted_chunk = ''.join(redacted_parts)
        redacted_chunks.append(redacted_chunk)
    merged_text = ''.join(redacted_chunks)
    merged_text = re.sub(r'(\[[A-Z]+\])(\s*\1)+', r'\1', merged_text)
    return merged_text


