import time 
import torch 
import torch .nn as nn 
import torch .optim as optim 
import pandas as pd 
import numpy as np 
import os 
from eda_preprocessing import run_eda_and_preprocessing 
def benchmark_cloud_training ():
    print ("="*70 )
    print ("   CLOUD COMPUTE & GPU ACCELERATION BENCHMARK (REQUIREMENT #11)")
    print ("="*70 )
    has_cuda =torch .cuda .is_available ()
    has_mps =hasattr (torch .backends ,'mps')and torch .backends .mps .is_available ()
    if has_cuda :
        device_type ='cuda'
        device_name =torch .cuda .get_device_name (0 )
    elif has_mps :
        device_type ='mps'
        device_name ='Apple Silicon GPU (MPS)'
    else :
        device_type ='cpu'
        device_name ='CPU System Execution'
    device =torch .device (device_type )
    print (f"Primary Compute Target : {device_type .upper ()}")
    print (f"Hardware Device Name   : {device_name }")
    print (f"CUDA Available         : {has_cuda }")
    print (f"MPS Available          : {has_mps }")
    print (f"PyTorch Version        : {torch .__version__ }")
    print ("\n--- Tensor Matrix Multiplication Benchmark ---")
    matrix_size =2000 
    iterations =50 
    cpu_x =torch .randn (matrix_size ,matrix_size ,device ='cpu')
    cpu_y =torch .randn (matrix_size ,matrix_size ,device ='cpu')
    _ =torch .matmul (cpu_x ,cpu_y )
    t0 =time .time ()
    for _ in range (iterations ):
        _ =torch .matmul (cpu_x ,cpu_y )
    cpu_time =time .time ()-t0 
    print (f"CPU MatMul ({matrix_size }x{matrix_size }, {iterations } iter) : {cpu_time :.4f} seconds")
    if has_cuda or has_mps :
        dev_x =torch .randn (matrix_size ,matrix_size ,device =device )
        dev_y =torch .randn (matrix_size ,matrix_size ,device =device )
        _ =torch .matmul (dev_x ,dev_y )
        if has_cuda :
            torch .cuda .synchronize ()
        t0 =time .time ()
        for _ in range (iterations ):
            _ =torch .matmul (dev_x ,dev_y )
        if has_cuda :
            torch .cuda .synchronize ()
        dev_time =time .time ()-t0 
        speedup =cpu_time /dev_time if dev_time >0 else 1.0 
        print (f"Accelerator MatMul ({device_type .upper ()})             : {dev_time :.4f} seconds ({speedup :.2f}x speedup)")
    else :
        print ("Accelerator MatMul (GPU/MPS)        : N/A (Colab T4 GPU recommended for large batch scaling)")
    print ("\n--- Deep Neural Network Training Batch Benchmark ---")
    _ ,X_train ,X_test ,y_train ,y_test ,X_train_scaled ,X_test_scaled ,scaler =run_eda_and_preprocessing ()
    X_tr_t =torch .tensor (X_train_scaled ,dtype =torch .float32 ).to (device )
    y_tr_t =torch .tensor (y_train .values ,dtype =torch .float32 ).unsqueeze (1 ).to (device )
    model =nn .Sequential (
    nn .Linear (48 ,128 ),nn .BatchNorm1d (128 ),nn .ReLU (),nn .Dropout (0.15 ),
    nn .Linear (128 ,64 ),nn .BatchNorm1d (64 ),nn .ReLU (),nn .Dropout (0.15 ),
    nn .Linear (64 ,32 ),nn .BatchNorm1d (32 ),nn .ReLU (),nn .Dropout (0.1 ),
    nn .Linear (32 ,16 ),nn .BatchNorm1d (16 ),nn .ReLU (),
    nn .Linear (16 ,8 ),nn .BatchNorm1d (8 ),nn .ReLU (),
    nn .Linear (8 ,1 ),nn .Sigmoid ()
    ).to (device )
    criterion =nn .BCELoss ()
    optimizer =optim .Adam (model .parameters (),lr =0.002 )
    batch_size =128 
    dataset_size =len (X_tr_t )
    num_batches =int (np .ceil (dataset_size /batch_size ))
    t0 =time .time ()
    model .train ()
    for epoch in range (10 ):
        permutation =torch .randperm (dataset_size )
        for i in range (0 ,dataset_size ,batch_size ):
            indices =permutation [i :i +batch_size ]
            bx ,by =X_tr_t [indices ],y_tr_t [indices ]
            optimizer .zero_grad ()
            out =model (bx )
            loss =criterion (out ,by )
            loss .backward ()
            optimizer .step ()
    if has_cuda :
        torch .cuda .synchronize ()
    bench_time =time .time ()-t0 
    samples_per_sec =(10 *dataset_size )/bench_time 
    print (f"Benchmark Epochs      : 10 Epochs ({10 *num_batches } total batch updates)")
    print (f"Benchmark Execution   : {bench_time :.4f} seconds")
    print (f"Training Throughput   : {samples_per_sec :.2f} samples/sec")
    print (f"Cloud Notebook        : colab_phishing_nn.ipynb configured for Google Colab T4 GPU")
    print ("="*70 )
if __name__ =="__main__":
    benchmark_cloud_training ()
