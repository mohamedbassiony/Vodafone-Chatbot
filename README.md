# Vodafone-Chatbot

## Project Overview

Vodafone Chatbot is a managerial-level simulation designed to streamline database interactions using Large Language Models (LLMs), Python and LangChain. This project leverages MySQL for data storage and retrieval, and utilizes LLMs to generate SQL queries based on user questions. The resulting data is then processed and, if it represents a time series, visualized using graphs to facilitate easy interpretation.

![Project pipeline with langchain and LLM](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-0YaK8SHX-0w-lvZFDNCnegesg0tYdBMOgw&s)

## Requirements

- Python 3.8 or later 

### Install Python using MiniConda

1) Download and install MiniConda From [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n vodafone-chatbot python=3.8
```

Another way to create a new environment is by using the following command:
```bash
$ conda create --name vodafone-chatbot python=3.8
```

3) Activate the environment:
```bash
$ conda activate vodafone-chatbot
```

### (Optional) work on powershell environment

1) Install Virtualenv
```bash
$ pip install virtualenv
```

2) Create a new environment using the following command:
```bash
$ python -m venv vodafone-chatbot
```

3) Activate the environment:
```bash
$ .\vodafone-chatbot\Scripts\Activate.ps1
```

### (Optional) Setup your command line for better readability
```bash
export PS1=export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages in src directory

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `API_KEYS` value.