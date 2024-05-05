import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import gc
import logging
import os

# Set environment variable to help with memory allocation
# os.environ["HF_HOME"] = "./hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
# os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
# os.environ["HF_METRICS_CACHE"] = "./hf_cache"
# os.environ["HF_MODULES_CACHE"] = "./hf_cache"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# os.environ["HF_TOKEN"] = "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"


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(lineno)s:%(message)s',
                    handlers=[logging.FileHandler('custom_log.log'),
                              logging.StreamHandler()]
                    )
logger = logging.getLogger(__name__)

def process_dataset(checkpoint, dataset, file_path):
    logger.info('in the function process_dataset.....')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    logger.info('initialized tokenizer...')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto",offload_folder='C:/Users/GPU/Documents/Phani/RSAIFinalProject')
    logger.info('initialized model...')
    #model.to(torch.device("cuda"))
    
    # Ensure that the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    prompt = """
    Please write the function based on the requirement.
    You must complete all code.
    The output must in triple backticks format script~(i.e., ```python ```).
    You should follow the following rules to write the function:
    First, avoid use print, try to use return.
    Second, do not write a machine learning model, try just a software function.
    """
    logger.info('DataSet Leng:%s '%(len(dataset)))
    #for i in tqdm(range(len(dataset) - 10)):
    for i in tqdm(range(len(dataset))):
        # inputs = tokenizer.encode(prompt + dataset[i]["prompt"], return_tensors="pt").to(model.device)
        encoded_input = tokenizer(prompt + dataset[i]["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = encoded_input['input_ids'].to(model.device)
        attention_mask = encoded_input['attention_mask'].to(model.device)
        
        logger.info('i value:%s inputs:%s'%(i, inputs))
        with torch.no_grad():  # Reduces memory usage
            logger.info('torch summary%s'%torch.cuda.memory_summary(device=None, abbreviated=False))
            # outputs = model.generate(inputs, max_new_tokens=256)
            outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=50)
            logger.info('torch summary%s'%torch.cuda.memory_summary(device=None, abbreviated=False))
        dataset[i]["completion"] = tokenizer.decode(outputs[0])
        logger.info('dataset[i] completion')
        torch.cuda.empty_cache()  # Frees up unused memory
        logger.info('clearing cuda cache.....')
    logger.info('\n\n for loop completed...')
    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=4)
    logger.info('dumped data in to file:%s'%file_path)
    del model  # Free up memory
    logger.info('deleted model....')
    torch.cuda.empty_cache()
    logger.info('cleared the cache....')

    # del variables
    # logger.info('deleted the variables........')
    gc.collect()
    logger.info('completed gc.collect')
# Load dataset
with open("./codellamaTest1.json", "r") as f:
    dataset = json.load(f)
logger.info('loaded dataset........ from ./codellamaTest1.json')

# Ignore ++

# Process dataset with each model
# process_dataset("codellama/CodeLlama-7b-hf", dataset, "./json_save/codellama.json")

# Ignore --

process_dataset("codellama/CodeLlama-7b-hf", dataset, "codellama.json")
logger.info('called process_dataset function..using llama-7b-hf')

# process_dataset("bigcode/starcoder", dataset, "./json_save/starcoder.json")

# process_dataset("bigcode/starcoder", dataset, "starcoder.json")
# logger.info('called process_dataset with bigcode/starcoder...')
