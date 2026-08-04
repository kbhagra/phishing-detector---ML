import os 
import joblib 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import TensorDataset ,DataLoader 
from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import accuracy_score ,classification_report ,confusion_matrix ,roc_auc_score 
from get_dataset import load_dataset 
torch .manual_seed (42 )
np .random .seed (42 )
class PhishingStandardLSTM (nn .Module ):
    """
    Standard (Unidirectional) Recurrent Neural Network (LSTM) for Phishing Detection.
    Processes feature inputs sequentially token-by-token (left-to-right) across 48 sequence timesteps.
    Architecture:
      - 2-Layer Standard Unidirectional LSTM (hidden_dim=64, bidirectional=False)
      - Dense Layer (64 -> 32) + ReLU + Dropout
      - Output Layer (32 -> 1) + Sigmoid Activation
    """
    def __init__ (self ,input_size =1 ,hidden_dim =64 ,num_layers =2 ,dropout_rate =0.2 ):
        super (PhishingStandardLSTM ,self ).__init__ ()
        self .lstm =nn .LSTM (
        input_size =input_size ,
        hidden_size =hidden_dim ,
        num_layers =num_layers ,
        batch_first =True ,
        dropout =dropout_rate if num_layers >1 else 0.0 ,
        bidirectional =False 
        )
        self .fc1 =nn .Linear (hidden_dim ,32 )
        self .relu =nn .ReLU ()
        self .dropout =nn .Dropout (dropout_rate )
        self .fc2 =nn .Linear (32 ,1 )
        self .sigmoid =nn .Sigmoid ()
    def forward (self ,x ):
        lstm_out ,(hn ,cn )=self .lstm (x )
        last_timestep_out =lstm_out [:,-1 ,:]
        x =self .dropout (self .relu (self .fc1 (last_timestep_out )))
        out =self .sigmoid (self .fc2 (x ))
        return out 
def train_lstm_model (epochs =20 ,batch_size =128 ,learning_rate =0.002 ,weight_decay =1e-4 ):
    print ("Loading dataset...")
    df ,csv_path =load_dataset ()
    X =df .drop (columns =['id','CLASS_LABEL'])
    y =df ['CLASS_LABEL']
    target_names =['Class 0: Malicious Phishing','Class 1: Legitimate']
    print (f"\nFeature matrix shape: {X .shape }")
    print (f"Target distribution:\n{y .value_counts ()}")
    print ("Class 0 -> Malicious Phishing Links")
    print ("Class 1 -> Legitimate Links")
    X_train ,X_test ,y_train ,y_test =train_test_split (
    X ,y ,test_size =0.2 ,random_state =42 ,stratify =y 
    )
    print (f"\nTrain set: {X_train .shape [0 ]} samples | Test set: {X_test .shape [0 ]} samples")
    scaler =StandardScaler ()
    X_train_scaled =scaler .fit_transform (X_train )
    X_test_scaled =scaler .transform (X_test )
    X_tr_seq =torch .tensor (X_train_scaled ,dtype =torch .float32 ).unsqueeze (2 )
    y_tr_t =torch .tensor (y_train .values ,dtype =torch .float32 ).unsqueeze (1 )
    X_te_seq =torch .tensor (X_test_scaled ,dtype =torch .float32 ).unsqueeze (2 )
    y_te_t =torch .tensor (y_test .values ,dtype =torch .float32 ).unsqueeze (1 )
    train_dataset =TensorDataset (X_tr_seq ,y_tr_t )
    train_loader =DataLoader (train_dataset ,batch_size =batch_size ,shuffle =True )
    model =PhishingStandardLSTM (input_size =1 ,hidden_dim =64 ,num_layers =2 )
    criterion =nn .BCELoss ()
    optimizer =optim .Adam (model .parameters (),lr =learning_rate ,weight_decay =weight_decay )
    scheduler =optim .lr_scheduler .ReduceLROnPlateau (optimizer ,mode ='min',factor =0.5 ,patience =3 )
    print (f"\n--- Standard Unidirectional LSTM Architecture ---")
    print (model )
    print (f"\nTraining Standard LSTM (Token-by-Token) over {epochs } epochs...")
    train_losses ,val_losses =[],[]
    train_accs ,val_accs =[],[]
    for epoch in range (1 ,epochs +1 ):
        model .train ()
        running_tr_loss =0.0 
        correct_tr =0 
        total_tr =0 
        for batch_X ,batch_y in train_loader :
            optimizer .zero_grad ()
            out =model (batch_X )
            loss =criterion (out ,batch_y )
            loss .backward ()
            optimizer .step ()
            running_tr_loss +=loss .item ()*batch_X .size (0 )
            preds =(out >=0.5 ).float ()
            correct_tr +=(preds ==batch_y ).sum ().item ()
            total_tr +=batch_X .size (0 )
        tr_loss =running_tr_loss /total_tr 
        tr_acc =(correct_tr /total_tr )*100 
        model .eval ()
        with torch .no_grad ():
            va_outputs =model (X_te_seq )
            va_loss =criterion (va_outputs ,y_te_t ).item ()
            va_preds =(va_outputs .numpy ()>=0.5 ).astype (int )
            va_acc =accuracy_score (y_test ,va_preds )*100 
        scheduler .step (va_loss )
        train_losses .append (tr_loss )
        val_losses .append (va_loss )
        train_accs .append (tr_acc )
        val_accs .append (va_acc )
        print (f"Epoch {epoch :2d}/{epochs } | "
        f"Train Loss: {tr_loss :.4f} | Val Loss: {va_loss :.4f} | "
        f"Train Acc: {tr_acc :.2f}% | Val Acc: {va_acc :.2f}%")
    model .eval ()
    with torch .no_grad ():
        final_probs =model (X_te_seq ).numpy ()
        final_preds =(final_probs >=0.5 ).astype (int )
    acc =accuracy_score (y_test ,final_preds )
    roc_auc =roc_auc_score (y_test ,final_probs )
    cm =confusion_matrix (y_test ,final_preds )
    report =classification_report (y_test ,final_preds ,target_names =target_names )
    print ("\n"+"="*60 )
    print ("      STANDARD UNIDIRECTIONAL LSTM RESULTS")
    print ("="*60 )
    print (f"Accuracy : {acc *100 :.2f}%")
    print (f"ROC-AUC  : {roc_auc :.4f}")
    print ("\nConfusion Matrix:")
    print (pd .DataFrame (cm ,index =target_names ,columns =['Pred 0 (Phishing)','Pred 1 (Legitimate)']))
    print ("\nClassification Report:")
    print (report )
    plots_dir ="plots"
    os .makedirs (plots_dir ,exist_ok =True )
    fig ,axes =plt .subplots (1 ,2 ,figsize =(14 ,5 ))
    axes [0 ].plot (range (1 ,epochs +1 ),train_losses ,label ='Train Loss',color ='#1f77b4',linewidth =2 )
    axes [0 ].plot (range (1 ,epochs +1 ),val_losses ,label ='Validation Loss',color ='#d62728',linewidth =2 ,linestyle ='--')
    axes [0 ].set_title ('Standard LSTM: Loss over Epochs',fontsize =12 ,fontweight ='bold')
    axes [0 ].set_xlabel ('Epoch',fontsize =10 )
    axes [0 ].set_ylabel ('Binary Cross-Entropy Loss',fontsize =10 )
    axes [0 ].legend (loc ='upper right')
    axes [0 ].grid (True ,linestyle =':',alpha =0.6 )
    axes [1 ].plot (range (1 ,epochs +1 ),train_accs ,label ='Train Accuracy',color ='#1f77b4',linewidth =2 )
    axes [1 ].plot (range (1 ,epochs +1 ),val_accs ,label ='Validation Accuracy',color ='#2ca02c',linewidth =2 ,linestyle ='--')
    axes [1 ].set_title ('Standard LSTM: Accuracy over Epochs',fontsize =12 ,fontweight ='bold')
    axes [1 ].set_xlabel ('Epoch',fontsize =10 )
    axes [1 ].set_ylabel ('Accuracy (%)',fontsize =10 )
    axes [1 ].legend (loc ='lower right')
    axes [1 ].grid (True ,linestyle =':',alpha =0.6 )
    plt .tight_layout ()
    curve_plot_path =os .path .join (plots_dir ,"lstm_curves.png")
    plt .savefig (curve_plot_path ,dpi =300 )
    plt .close ()
    print (f"\nSaved training curves plot to      : {curve_plot_path }")
    plt .figure (figsize =(6 ,5 ))
    plt .imshow (cm ,interpolation ='nearest',cmap =plt .cm .Oranges )
    plt .title ('Standard LSTM Model: Confusion Matrix',fontsize =12 ,fontweight ='bold')
    plt .colorbar ()
    tick_marks =np .arange (len (target_names ))
    plt .xticks (tick_marks ,['Phishing (0)','Legitimate (1)'],rotation =0 )
    plt .yticks (tick_marks ,['Phishing (0)','Legitimate (1)'])
    thresh =cm .max ()/2. 
    for i in range (cm .shape [0 ]):
        for j in range (cm .shape [1 ]):
            plt .text (j ,i ,format (cm [i ,j ],'d'),
            horizontalalignment ="center",
            color ="white"if cm [i ,j ]>thresh else "black",
            fontsize =12 ,fontweight ='bold')
    plt .ylabel ('True Class')
    plt .xlabel ('Predicted Class')
    plt .tight_layout ()
    cm_plot_path =os .path .join (plots_dir ,"lstm_confusion_matrix.png")
    plt .savefig (cm_plot_path ,dpi =300 )
    plt .close ()
    print (f"Saved confusion matrix plot to     : {cm_plot_path }")
    models_dir ="models"
    os .makedirs (models_dir ,exist_ok =True )
    model_path =os .path .join (models_dir ,"lstm_model.pt")
    scaler_path =os .path .join (models_dir ,"lstm_scaler.pt")
    torch .save (model .state_dict (),model_path )
    joblib .dump (scaler ,scaler_path )
    print (f"Standard LSTM Model weights saved to: {model_path }")
    print (f"Standard LSTM Scaler saved to       : {scaler_path }")
    print ("="*60 )
    return model ,scaler 
if __name__ =="__main__":
    train_lstm_model (epochs =20 )
