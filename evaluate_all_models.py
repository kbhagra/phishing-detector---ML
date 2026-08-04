import os 
import joblib 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import (
accuracy_score ,precision_score ,recall_score ,f1_score ,
roc_auc_score ,roc_curve ,confusion_matrix 
)
import torch 
from eda_preprocessing import run_eda_and_preprocessing 
from mlp_neural_network import PhishingReLU_MLP 
from lstm_model import PhishingStandardLSTM 
from raw_url_char_lstm import CharLevelURLLSTM ,tokenize_url ,generate_urls_from_features ,VOCAB_SIZE 
def evaluate_all_models ():
    print ("="*80 )
    print ("      MASTER HELD-OUT TEST EVALUATION & METRICS COMPARISON (REQ #9)")
    print ("="*80 )
    df ,X_train ,X_test ,y_train ,y_test ,X_train_scaled ,X_test_scaled ,scaler =run_eda_and_preprocessing ()
    models_dir ="models"
    plots_dir ="plots"
    os .makedirs (plots_dir ,exist_ok =True )
    results =[]
    roc_data ={}
    def record_metrics (name ,y_true ,y_pred ,y_prob ):
        acc =accuracy_score (y_true ,y_pred )
        prec =precision_score (y_true ,y_pred )
        rec =recall_score (y_true ,y_pred )
        f1 =f1_score (y_true ,y_pred )
        auc =roc_auc_score (y_true ,y_prob )
        fpr ,tpr ,_ =roc_curve (y_true ,y_prob )
        roc_data [name ]=(fpr ,tpr ,auc )
        results .append ({
        'Model Family':name ,
        'Accuracy (%)':f"{acc *100 :.2f}%",
        'Precision':f"{prec :.4f}",
        'Recall':f"{rec :.4f}",
        'F1-Score':f"{f1 :.4f}",
        'ROC-AUC':f"{auc :.4f}"
        })
        print (f"Loaded & Evaluated {name :32s} | Acc: {acc *100 :.2f}% | AUC: {auc :.4f}")
    maj_model =joblib .load (os .path .join (models_dir ,"majority_baseline.pt"))
    maj_pred =maj_model .predict (X_test )
    maj_prob =maj_model .predict_proba (X_test )[:,1 ]
    record_metrics ("Baseline: Majority Classifier",y_test ,maj_pred ,maj_prob )
    dt_model =joblib .load (os .path .join (models_dir ,"decision_tree_baseline.pt"))
    dt_pred =dt_model .predict (X_test )
    dt_prob =dt_model .predict_proba (X_test )[:,1 ]
    record_metrics ("Baseline: Decision Tree Stump",y_test ,dt_pred ,dt_prob )
    lr_model =joblib .load (os .path .join (models_dir ,"logistic_regression_model.pt"))
    lr_pred =lr_model .predict (X_test_scaled )
    lr_prob =lr_model .predict_proba (X_test_scaled )[:,1 ]
    record_metrics ("Linear: Logistic Regression",y_test ,lr_pred ,lr_prob )
    rf_model =joblib .load (os .path .join (models_dir ,"random_forest_model.pt"))
    rf_pred =rf_model .predict (X_test )
    rf_prob =rf_model .predict_proba (X_test )[:,1 ]
    record_metrics ("Tree-Based: Random Forest",y_test ,rf_pred ,rf_prob )
    mlp =PhishingReLU_MLP (input_dim =X_train .shape [1 ])
    mlp .load_state_dict (torch .load (os .path .join (models_dir ,"mlp_model.pt")))
    mlp .eval ()
    X_test_t =torch .tensor (X_test_scaled ,dtype =torch .float32 )
    with torch .no_grad ():
        mlp_prob =mlp (X_test_t ).numpy ().flatten ()
        mlp_pred =(mlp_prob >=0.5 ).astype (int )
    record_metrics ("Deep NN: 5-Layer ReLU MLP",y_test ,mlp_pred ,mlp_prob )
    lstm =PhishingStandardLSTM (input_size =1 ,hidden_dim =64 ,num_layers =2 )
    lstm .load_state_dict (torch .load (os .path .join (models_dir ,"lstm_model.pt")))
    lstm .eval ()
    X_test_seq =torch .tensor (X_test_scaled ,dtype =torch .float32 ).unsqueeze (2 )
    with torch .no_grad ():
        lstm_prob =lstm (X_test_seq ).numpy ().flatten ()
        lstm_pred =(lstm_prob >=0.5 ).astype (int )
    record_metrics ("RNN: Standard Feature LSTM",y_test ,lstm_pred ,lstm_prob )
    if os .path .exists (os .path .join (models_dir ,"raw_url_char_lstm.pt")):
        char_lstm =CharLevelURLLSTM (vocab_size =VOCAB_SIZE ,embed_dim =32 ,hidden_dim =64 ,num_layers =2 )
        char_lstm .load_state_dict (torch .load (os .path .join (models_dir ,"raw_url_char_lstm.pt")))
        char_lstm .eval ()
        url_strings =generate_urls_from_features (df )
        labels =df ['CLASS_LABEL'].values 
        tokenized_data =[tokenize_url (url ,max_length =120 )for url in url_strings ]
        X_tokenized =np .array ([t [0 ]for t in tokenized_data ])
        X_lengths =np .array ([t [1 ]for t in tokenized_data ])
        _ ,X_te_tok ,_ ,len_te ,_ ,y_te_raw =train_test_split (
        X_tokenized ,X_lengths ,labels ,test_size =0.2 ,random_state =42 ,stratify =labels 
        )
        X_te_t_char =torch .tensor (X_te_tok ,dtype =torch .long )
        len_te_t_char =torch .tensor (len_te ,dtype =torch .long )
        with torch .no_grad ():
            char_prob =char_lstm (X_te_t_char ,len_te_t_char ).numpy ().flatten ()
            char_pred =(char_prob >=0.5 ).astype (int )
        record_metrics ("RNN: Character-Level URL LSTM",y_te_raw ,char_pred ,char_prob )
    res_df =pd .DataFrame (results )
    print ("\n"+"="*80 )
    print ("      FINAL HELD-OUT TEST EVALUATION SUMMARY TABLE")
    print ("="*80 )
    print (res_df .to_string (index =False ))
    print ("="*80 )
    res_df .to_csv (os .path .join (plots_dir ,"master_evaluation_metrics.csv"),index =False )
    plt .figure (figsize =(9 ,7 ))
    palette =['#7f7f7f','#bcbd22','#1f77b4','#2ca02c','#d62728','#9467bd','#e377c2']
    for idx ,(name ,(fpr ,tpr ,auc ))in enumerate (roc_data .items ()):
        color =palette [idx %len (palette )]
        linestyle ='--'if 'Baseline'in name else '-'
        linewidth =2.5 if 'Random Forest'in name or 'MLP'in name else 1.8 
        plt .plot (fpr ,tpr ,label =f"{name } (AUC = {auc :.4f})",color =color ,linestyle =linestyle ,linewidth =linewidth )
    plt .plot ([0 ,1 ],[0 ,1 ],'k:',label ='Random Chance (AUC = 0.5000)',alpha =0.7 )
    plt .xlim ([-0.01 ,1.0 ])
    plt .ylim ([0.0 ,1.02 ])
    plt .xlabel ('False Positive Rate (1 - Specificity)',fontsize =11 )
    plt .ylabel ('True Positive Rate (Sensitivity / Recall)',fontsize =11 )
    plt .title ('Master Receiver Operating Characteristic (ROC) Curves Across All Models',fontsize =12 ,fontweight ='bold')
    plt .grid (True ,linestyle =':',alpha =0.6 )
    plt .legend (loc ='lower right',fontsize =9 )
    plt .tight_layout ()
    roc_plot_path =os .path .join (plots_dir ,"all_models_roc_curves.png")
    plt .savefig (roc_plot_path ,dpi =300 )
    plt .close ()
    print (f"\nSaved Master ROC curves plot to: {roc_plot_path }")
    print ("="*80 )
    return res_df 
if __name__ =="__main__":
    evaluate_all_models ()
