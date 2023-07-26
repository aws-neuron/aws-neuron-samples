import os
import time
import torch
from transformers import AutoTokenizer
from transformers_neuronx.gptj.model import GPTJForSampling
from multiprocessing import Process

def load_model_infer():
    # load model to NeuronCores with 8-way tensor parallel and DP
    load_compile_time = time.time()
    neuron_model = GPTJForSampling.from_pretrained('./gptj-6b-split', n_positions=1024, batch_size=64, tp_degree=8, amp='f16')
    neuron_model.to_neuron()
    load_compile_elapsed = time.time() - load_compile_time
    print(f'Model load & compile time in a single process  {load_compile_elapsed} seconds')

    # construct a tokenizer and encode prompt text
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    batch_prompts = [
        "I am specialized at sentence generation language models,", 
    ]
    batch_prompts = batch_prompts * 64

    input_ids = torch.as_tensor([tokenizer.encode(text) for text in batch_prompts])


    with torch.inference_mode():
        # warmup
        generated_sequences = neuron_model.sample(input_ids, sequence_length=1024)
        
        start = time.time()
        for i in range(2):
            generated_sequences = neuron_model.sample(input_ids, sequence_length=1024)
        elapsed = (time.time() - start) / 2

        generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'Averaged Latency for one inference {elapsed} seconds')

if __name__ == '__main__':
    os.environ['NEURON_RT_NUM_CORES']='8'
    total_start = time.time()
    p1 = Process(target=load_model_infer)
    p2 = Process(target=load_model_infer)
    p3 = Process(target=load_model_infer)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    total_elapsed = time.time() - total_start
    print(f'total processes time including compilation finished in {total_elapsed} seconds')
    print(f'TPS {(30/total_elapsed)*64} ')
    p1.terminate()
    p2.terminate()
    p3.terminate()