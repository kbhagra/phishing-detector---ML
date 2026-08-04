import os 
import warnings 
import joblib 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import TensorDataset ,DataLoader 
from sklearn .model_selection import StratifiedKFold 
from sklearn .preprocessing import StandardScaler 
from sklearn .linear_model import LogisticRegression 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .metrics import accuracy_score 
from get_dataset import load_dataset 
warnings .filterwarnings ('ignore')
np .random .seed (42 )
torch .manual_seed (42 )
def run_logistic_regression_tuning (X_scaled ,y ,cv ):
    print ("\n"+"="*70 )
    print ("  1. LOGISTIC REGRESSION: L1 (LASSO) VS L2 (RIDGE) & C REGULARIZATION")
    print ("="*70 )
    C_values =[0.001 ,0.01 ,0.1 ,1.0 ,10.0 ,100.0 ]
    penalties =['l1','l2']
    results =[]
    for penalty in penalties :
        for C in C_values :
            lr =LogisticRegression (penalty =penalty ,C =C ,solver ='liblinear',random_state =42 ,max_iter =1000 )
            train_scores ,val_scores =[],[]
            num_zero_features =[]
            for train_idx ,val_idx in cv .split (X_scaled ,y ):
                X_tr ,X_va =X_scaled [train_idx ],X_scaled [val_idx ]
                y_tr ,y_va =y .iloc [train_idx ],y .iloc [val_idx ]
                lr .fit (X_tr ,y_tr )
                tr_acc =accuracy_score (y_tr ,lr .predict (X_tr ))
                va_acc =accuracy_score (y_va ,lr .predict (X_va ))
                train_scores .append (tr_acc )
                val_scores .append (va_acc )
                num_zero_features .append (np .sum (lr .coef_ ==0 ))
            mean_tr_acc =np .mean (train_scores )*100 
            mean_va_acc =np .mean (val_scores )*100 
            mean_zeros =np .mean (num_zero_features )
            overfit_gap =mean_tr_acc -mean_va_acc 
            results .append ({
            'penalty':penalty .upper (),
            'C':C ,
            'train_acc':mean_tr_acc ,
            'val_acc':mean_va_acc ,
            'overfit_gap':overfit_gap ,
            'zero_features':mean_zeros 
            })
            print (f"Penalty: {penalty .upper ():2s} | C: {C :7.3f} | Train Acc: {mean_tr_acc :.2f}% | "
            f"Val Acc: {mean_va_acc :.2f}% | Gap: {overfit_gap :.2f}% | Zero Features: {mean_zeros :.1f}/48")
    res_df =pd .DataFrame (results )
    best_lr_row =res_df .loc [res_df ['val_acc'].idxmax ()]
    print (f"\n---> Best Logistic Regression Config: Penalty={best_lr_row ['penalty']}, C={best_lr_row ['C']} "
    f"with Val Accuracy={best_lr_row ['val_acc']:.2f}%")
    return res_df ,best_lr_row 
def run_random_forest_tuning (X ,y ,cv ):
    print ("\n"+"="*70 )
    print ("  2. RANDOM FOREST: TREE DEPTH & REGULARIZATION SEARCH")
    print ("="*70 )
    depths =[3 ,5 ,8 ,12 ,20 ,None ]
    results =[]
    for depth in depths :
        rf =RandomForestClassifier (n_estimators =100 ,max_depth =depth ,min_samples_split =5 ,random_state =42 ,n_jobs =-1 )
        train_scores ,val_scores =[],[]
        for train_idx ,val_idx in cv .split (X ,y ):
            X_tr ,X_va =X .iloc [train_idx ],X .iloc [val_idx ]
            y_tr ,y_va =y .iloc [train_idx ],y .iloc [val_idx ]
            rf .fit (X_tr ,y_tr )
            train_scores .append (accuracy_score (y_tr ,rf .predict (X_tr )))
            val_scores .append (accuracy_score (y_va ,rf .predict (X_va )))
        mean_tr_acc =np .mean (train_scores )*100 
        mean_va_acc =np .mean (val_scores )*100 
        gap =mean_tr_acc -mean_va_acc 
        depth_str =str (depth )if depth is not None else "None (Unconstrained)"
        results .append ({
        'max_depth':depth_str ,
        'train_acc':mean_tr_acc ,
        'val_acc':mean_va_acc ,
        'overfit_gap':gap 
        })
        print (f"Max Depth: {depth_str :20s} | Train Acc: {mean_tr_acc :.2f}% | Val Acc: {mean_va_acc :.2f}% | Gap: {gap :.2f}%")
    res_df =pd .DataFrame (results )
    best_rf_row =res_df .loc [res_df ['val_acc'].idxmax ()]
    print (f"\n---> Best Random Forest Config: Max Depth={best_rf_row ['max_depth']} "
    f"with Val Accuracy={best_rf_row ['val_acc']:.2f}%")
    return res_df ,best_rf_row 
class ConfigurableMLP (nn .Module ):
    def __init__ (self ,input_dim =48 ,dropout_rate =0.2 ):
        super (ConfigurableMLP ,self ).__init__ ()
        self .net =nn .Sequential (
        nn .Linear (input_dim ,128 ),
        nn .BatchNorm1d (128 ),
        nn .ReLU (),
        nn .Dropout (dropout_rate ),
        nn .Linear (128 ,64 ),
        nn .BatchNorm1d (64 ),
        nn .ReLU (),
        nn .Dropout (dropout_rate ),
        nn .Linear (64 ,32 ),
        nn .BatchNorm1d (32 ),
        nn .ReLU (),
        nn .Linear (32 ,16 ),
        nn .BatchNorm1d (16 ),
        nn .ReLU (),
        nn .Linear (16 ,8 ),
        nn .BatchNorm1d (8 ),
        nn .ReLU (),
        nn .Linear (8 ,1 ),
        nn .Sigmoid ()
        )
    def forward (self ,x ):
        return self .net (x )
def run_mlp_regularization_tuning (X_scaled ,y ,cv ):
    print ("\n"+"="*70 )
    print ("  3. NEURAL NETWORK (MLP): DROPOUT & L2 WEIGHT DECAY REGULARIZATION")
    print ("="*70 )
    dropout_rates =[0.0 ,0.1 ,0.25 ,0.4 ]
    weight_decays =[0.0 ,1e-4 ,1e-3 ,1e-2 ]
    results =[]
    X_t =torch .tensor (X_scaled ,dtype =torch .float32 )
    y_t =torch .tensor (y .values ,dtype =torch .float32 ).unsqueeze (1 )
    train_idx ,val_idx =next (cv .split (X_scaled ,y ))
    X_tr ,y_tr =X_t [train_idx ],y_t [train_idx ]
    X_va ,y_va =X_t [val_idx ],y_t [val_idx ]
    loader =DataLoader (TensorDataset (X_tr ,y_tr ),batch_size =256 ,shuffle =True )
    for drop in dropout_rates :
        for wd in weight_decays :
            torch .manual_seed (42 )
            model =ConfigurableMLP (input_dim =48 ,dropout_rate =drop )
            criterion =nn .BCELoss ()
            optimizer =optim .Adam (model .parameters (),lr =0.003 ,weight_decay =wd )
            for epoch in range (15 ):
                model .train ()
                for b_x ,b_y in loader :
                    optimizer .zero_grad ()
                    loss =criterion (model (b_x ),b_y )
                    loss .backward ()
                    optimizer .step ()
            model .eval ()
            with torch .no_grad ():
                tr_out =model (X_tr )
                va_out =model (X_va )
                va_loss =criterion (va_out ,y_va ).item ()
                tr_pred =(tr_out .numpy ()>=0.5 ).astype (int )
                va_pred =(va_out .numpy ()>=0.5 ).astype (int )
                tr_acc =accuracy_score (y .iloc [train_idx ],tr_pred )*100 
                va_acc =accuracy_score (y .iloc [val_idx ],va_pred )*100 
                gap =tr_acc -va_acc 
            results .append ({
            'dropout':drop ,
            'weight_decay':wd ,
            'train_acc':tr_acc ,
            'val_acc':va_acc ,
            'val_loss':va_loss ,
            'overfit_gap':gap 
            })
            print (f"Dropout: {drop :4.2f} | Weight Decay (L2): {wd :6.4f} | Train Acc: {tr_acc :.2f}% | "
            f"Val Acc: {va_acc :.2f}% | Val Loss: {va_loss :.4f} | Gap: {gap :.2f}%")
    res_df =pd .DataFrame (results )
    best_mlp_row =res_df .loc [res_df ['val_acc'].idxmax ()]
    print (f"\n---> Best MLP Config: Dropout={best_mlp_row ['dropout']}, Weight Decay={best_mlp_row ['weight_decay']} "
    f"with Val Accuracy={best_mlp_row ['val_acc']:.2f}%")
    return res_df ,best_mlp_row 
def plot_all_tuning_results (lr_df ,rf_df ,mlp_df ):
    plots_dir ="plots"
    os .makedirs (plots_dir ,exist_ok =True )
    plt .figure (figsize =(8 ,5 ))
    l1_data =lr_df [lr_df ['penalty']=='L1']
    l2_data =lr_df [lr_df ['penalty']=='L2']
    plt .plot (l1_data ['C'],l1_data ['val_acc'],marker ='o',label ='L1 (Lasso) Val Acc',color ='#1f77b4',linewidth =2 )
    plt .plot (l2_data ['C'],l2_data ['val_acc'],marker ='s',label ='L2 (Ridge) Val Acc',color ='#2ca02c',linewidth =2 ,linestyle ='--')
    plt .xscale ('log')
    plt .xlabel ('Regularization Parameter C (Log Scale)',fontsize =10 )
    plt .ylabel ('5-Fold CV Accuracy (%)',fontsize =10 )
    plt .title ('Logistic Regression: Effect of Regularization C (L1 vs L2)',fontsize =12 ,fontweight ='bold')
    plt .grid (True ,linestyle =':',alpha =0.6 )
    plt .legend (loc ='lower right')
    plt .tight_layout ()
    plt .savefig (os .path .join (plots_dir ,"lr_regularization_C.png"),dpi =300 )
    plt .close ()
    plt .figure (figsize =(8 ,5 ))
    plt .plot (rf_df ['max_depth'],rf_df ['train_acc'],marker ='o',label ='Train Accuracy',color ='#1f77b4',linewidth =2 )
    plt .plot (rf_df ['max_depth'],rf_df ['val_acc'],marker ='s',label ='Validation Accuracy',color ='#d62728',linewidth =2 ,linestyle ='--')
    plt .xlabel ('Maximum Tree Depth (max_depth)',fontsize =10 )
    plt .ylabel ('5-Fold CV Accuracy (%)',fontsize =10 )
    plt .title ('Random Forest: Effect of Tree Depth Regularization',fontsize =12 ,fontweight ='bold')
    plt .grid (True ,linestyle =':',alpha =0.6 )
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (plots_dir ,"rf_depth_regularization.png"),dpi =300 )
    plt .close ()
    plt .figure (figsize =(8 ,5 ))
    for drop in [0.0 ,0.1 ,0.25 ,0.4 ]:
        sub =mlp_df [mlp_df ['dropout']==drop ]
        plt .plot (sub ['weight_decay'],sub ['val_acc'],marker ='o',label =f'Dropout = {drop }',linewidth =2 )
    plt .xscale ('symlog',linthresh =1e-4 )
    plt .xlabel ('Weight Decay L2 Penalty (Log Scale)',fontsize =10 )
    plt .ylabel ('Validation Accuracy (%)',fontsize =10 )
    plt .title ('MLP Neural Net: Effect of Dropout & L2 Regularization',fontsize =12 ,fontweight ='bold')
    plt .grid (True ,linestyle =':',alpha =0.6 )
    plt .legend ()
    plt .tight_layout ()
    plt .savefig (os .path .join (plots_dir ,"mlp_dropout_tuning.png"),dpi =300 )
    plt .close ()
    old_knn_plot =os .path .join (plots_dir ,"knn_k_tuning.png")
    if os .path .exists (old_knn_plot ):
        os .remove (old_knn_plot )
    print ("\nSaved tuning plots (Logistic Regression, Random Forest, MLP) to ./plots/")
def main ():
    print ("Loading dataset...")
    df ,csv_path =load_dataset ()
    X =df .drop (columns =['id','CLASS_LABEL'])
    y =df ['CLASS_LABEL']
    scaler =StandardScaler ()
    X_scaled =scaler .fit_transform (X )
    cv =StratifiedKFold (n_splits =5 ,shuffle =True ,random_state =42 )
    lr_df ,best_lr =run_logistic_regression_tuning (X_scaled ,y ,cv )
    rf_df ,best_rf =run_random_forest_tuning (X ,y ,cv )
    mlp_df ,best_mlp =run_mlp_regularization_tuning (X_scaled ,y ,cv )
    plot_all_tuning_results (lr_df ,rf_df ,mlp_df )
    print ("\n"+"="*85 )
    print ("      FINAL HYPERPARAMETER TUNING & REGULARIZATION REPORT SUMMARY")
    print ("="*85 )
    summary_data =[
    {"Model":"Logistic Regression","Best Config / Regularization":f"Penalty={best_lr ['penalty']}, C={best_lr ['C']}","Train Acc":f"{best_lr ['train_acc']:.2f}%","Val Acc":f"{best_lr ['val_acc']:.2f}%","Train-Val Gap":f"{best_lr ['overfit_gap']:.2f}%"},
    {"Model":"Random Forest","Best Config / Regularization":f"Max Depth={best_rf ['max_depth']}","Train Acc":f"{best_rf ['train_acc']:.2f}%","Val Acc":f"{best_rf ['val_acc']:.2f}%","Train-Val Gap":f"{best_rf ['overfit_gap']:.2f}%"},
    {"Model":"5-Layer MLP Neural Net","Best Config / Regularization":f"Dropout={best_mlp ['dropout']}, L2={best_mlp ['weight_decay']}","Train Acc":f"{best_mlp ['train_acc']:.2f}%","Val Acc":f"{best_mlp ['val_acc']:.2f}%","Train-Val Gap":f"{best_mlp ['overfit_gap']:.2f}%"}
    ]
    summary_table =pd .DataFrame (summary_data )
    print (summary_table .to_string (index =False ))
    print ("="*85 )
if __name__ =="__main__":
    main ()
