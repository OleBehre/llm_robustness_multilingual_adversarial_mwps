# Multilingual, Adversarial Math Word Problems: Testing the Robustness of Large Language Models

## Table of Contents

- [Overview](#overview)  
- [HuggingFace Collection](#huggingface-collection)  
- [Dataset](#dataset)  
- [Benchmarking](#benchmarking)  
  - [Configurations](#configurations)  
  - [Benchmarking Scripts](#benchmarking-scripts)  
  - [Results](#results)


## Overview

This repository contains work for my bachelor thesis focused on **multilingual, adversarial math word problems**. The goal is to evaluate and test the robustness of Large Language Models (LLMs) and focused on their mathematical reasoning abilities. 

## HuggingFace Collection

A curated HuggingFace collection is available here:  
[**OleBehre's Bachelor Thesis Collection**](https://huggingface.co/collections/OleBehre/ole-behre-bachelor-thesis-677ec94123133233794a0ea8)

It includes all selected models used in this thesis.

## Dataset

The novel **MultiGSM-Adv** dataset is located in the [`01_dataset/`](01_dataset/) folder.  
This dataset is designed to test LLMs on adversarial math word problems in multiple languages. It translates [**GSM8k**](https://github.com/openai/grade-school-math) and [**GSM-Adv**](https://github.com/him1411/problemathic/tree/main) into eleven selected languages.

## Benchmarking

For all benchmarking the [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) is used.

### Configurations

Annotated YAML configuration files used for benchmarking can be found in the [`02_configurations/`](02_configurations/) folder.

### Benchmarking Scripts

All benchmarking scripts based on are uploaded in the [`03_benchmarking/`](03_benchmarking/) folder.

### Results

- Summarized results as CSV files: [`04_results/`](04_results/)  
- Raw output from `lm-eval`: [`99_archive_resultsjsons/`](99_archive_resultjsons/)

The archived JSONs contain the unprocessed evaluation metrics from each run.

