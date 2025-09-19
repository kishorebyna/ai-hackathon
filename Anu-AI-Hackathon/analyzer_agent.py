import os
import pandas

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


def read_sample_data(csvfile):
    
