import os 
import joblib 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import StandardScaler 
from sklearn .linear_model import SGDClassifier ,LogisticRegression 
from sklearn .metrics import log_loss ,accuracy_score ,classification_report ,confusion_matrix ,roc_auc_score 
from get_dataset import load_dataset 
def train_logistic_regression (epochs =100 ):
    print ("Loading dataset...")
    df ,csv_path =load_dataset ()
    X =df .drop (columns =['id','CLASS_LABEL'])
    y =df ['CLASS_LABEL']
    target_names =['Class 0: Malicious Phishing','Class 1: Legitimate']
    classes =np .unique (y )
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
    print (f"\nTraining Logistic Regression over {epochs } epochs...")
    model =SGDClassifier (loss ='log_loss',alpha =1e-4 ,random_state =42 )
    train_losses ,val_losses =[],[]
    train_accs ,val_accs =[],[]
    for epoch in range (1 ,epochs +1 ):
        model .partial_fit (X_train_scaled ,y_train ,classes =classes )
        train_prob =model .predict_proba (X_train_scaled )
        val_prob =model .predict_proba (X_test_scaled )
        train_pred =model .predict (X_train_scaled )
        val_pred =model .predict (X_test_scaled )
        tr_loss =log_loss (y_train ,train_prob )
        va_loss =log_loss (y_test ,val_prob )
        tr_acc =accuracy_score (y_train ,train_pred )
        va_acc =accuracy_score (y_test ,val_pred )
        train_losses .append (tr_loss )
        val_losses .append (va_loss )
        train_accs .append (tr_acc *100 )
        val_accs .append (va_acc *100 )
        if epoch ==1 or epoch %10 ==0 or epoch ==epochs :
            print (f"Epoch {epoch :3d}/{epochs } | "
            f"Train Loss: {tr_loss :.4f} | Val Loss: {va_loss :.4f} | "
            f"Train Acc: {tr_acc *100 :.2f}% | Val Acc: {va_acc *100 :.2f}%")
    y_pred =model .predict (X_test_scaled )
    y_prob =model .predict_proba (X_test_scaled )[:,1 ]
    acc =accuracy_score (y_test ,y_pred )
    roc_auc =roc_auc_score (y_test ,y_prob )
    cm =confusion_matrix (y_test ,y_pred )
    report =classification_report (y_test ,y_pred ,target_names =target_names )
    print ("\n"+"="*60 )
    print ("      FINAL LOGISTIC REGRESSION RESULTS")
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
    axes [0 ].set_title ('Logistic Regression: Loss over Epochs',fontsize =12 ,fontweight ='bold')
    axes [0 ].set_xlabel ('Epoch',fontsize =10 )
    axes [0 ].set_ylabel ('Log Loss (Binary Cross-Entropy)',fontsize =10 )
    axes [0 ].legend (loc ='upper right')
    axes [0 ].grid (True ,linestyle =':',alpha =0.6 )
    axes [1 ].plot (range (1 ,epochs +1 ),train_accs ,label ='Train Accuracy',color ='#1f77b4',linewidth =2 )
    axes [1 ].plot (range (1 ,epochs +1 ),val_accs ,label ='Validation Accuracy',color ='#2ca02c',linewidth =2 ,linestyle ='--')
    axes [1 ].set_title ('Logistic Regression: Accuracy over Epochs',fontsize =12 ,fontweight ='bold')
    axes [1 ].set_xlabel ('Epoch',fontsize =10 )
    axes [1 ].set_ylabel ('Accuracy (%)',fontsize =10 )
    axes [1 ].legend (loc ='lower right')
    axes [1 ].grid (True ,linestyle =':',alpha =0.6 )
    plt .tight_layout ()
    curve_plot_path =os .path .join (plots_dir ,"logistic_regression_curves.png")
    plt .savefig (curve_plot_path ,dpi =300 )
    plt .close ()
    print (f"\nSaved training curves plot to : {curve_plot_path }")
    plt .figure (figsize =(6 ,5 ))
    plt .imshow (cm ,interpolation ='nearest',cmap =plt .cm .Blues )
    plt .title ('Confusion Matrix',fontsize =12 ,fontweight ='bold')
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
    cm_plot_path =os .path .join (plots_dir ,"confusion_matrix.png")
    plt .savefig (cm_plot_path ,dpi =300 )
    plt .close ()
    print (f"Saved confusion matrix plot to: {cm_plot_path }")
    models_dir ="models"
    os .makedirs (models_dir ,exist_ok =True )
    model_path =os .path .join (models_dir ,"logistic_regression_model.pt")
    scaler_path =os .path .join (models_dir ,"scaler.pt")
    joblib .dump (model ,model_path )
    joblib .dump (scaler ,scaler_path )
    print (f"Model saved to                 : {model_path }")
    print (f"Scaler saved to                : {scaler_path }")
    print ("="*60 )
    return model ,scaler 
if __name__ =="__main__":
    train_logistic_regression (epochs =100 )
