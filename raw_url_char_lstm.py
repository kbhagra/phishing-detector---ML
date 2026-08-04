import os 
import string 
import joblib 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import torch 
import torch .nn as nn 
import torch .optim as optim 
from torch .utils .data import TensorDataset ,DataLoader 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import accuracy_score ,classification_report ,confusion_matrix ,roc_auc_score 
from get_dataset import load_dataset 
torch .manual_seed (42 )
np .random .seed (42 )
CHARSET =string .ascii_lowercase +string .digits +".-/:?=@_~#%&+"
CHAR_TO_ID ={c :i +1 for i ,c in enumerate (CHARSET )}
ID_TO_CHAR ={i +1 :c for i ,c in enumerate (CHARSET )}
VOCAB_SIZE =len (CHAR_TO_ID )+1 
def tokenize_url (url_string ,max_length =120 ):
    """
    Tokenizes a raw URL string character-by-character into integer token IDs.
    Pads or truncates to max_length.
    """
    cleaned_url =str (url_string ).lower ().strip ()
    token_ids =[CHAR_TO_ID .get (char ,0 )for char in cleaned_url [:max_length ]]
    seq_len =max (1 ,len (token_ids ))
    if len (token_ids )<max_length :
        token_ids +=[0 ]*(max_length -len (token_ids ))
    return token_ids ,seq_len 
def generate_urls_from_features (df ):
    """
    Synthesizes raw URL text strings matching row properties in dataset.
    Class 0 = Malicious Phishing Links
    Class 1 = Legitimate Links
    """
    urls =[]
    np .random .seed (42 )
    phish_brands =['paypal','bankofamerica','chase','wellsfargo','appleid','microsoft','netflix','amazon']
    phish_words =['login','verify','account-update','secure-confirm','banking-portal','signin','auth-check']
    legit_domains =['google.com','wikipedia.org','github.com','amazon.com','microsoft.com','yahoo.com','apple.com','linkedin.com']
    phish_tlds =['.xyz','.top','.info','.cc','.site','.club','.online','.work']
    legit_queries =['?q=search','?id=102','?ref=main','?category=tech','?lang=en','?view=full']
    for idx ,row in df .iterrows ():
        is_legitimate =(row ['CLASS_LABEL']==1 )
        protocol ="https://"if row .get ('NoHttps',0 )==0 else "http://"
        if not is_legitimate :
            if row .get ('IpAddress',0 )==1 :
                host =f"192.168.{np .random .randint (1 ,254 )}.{np .random .randint (1 ,254 )}"
            else :
                brand =np .random .choice (phish_brands )
                word =np .random .choice (phish_words )
                tld =np .random .choice (phish_tlds )
                sub ="sub."*int (min (row .get ('SubdomainLevel',1 ),2 ))
                host =f"{sub }{brand }-{word }{tld }"
            path_depth =int (min (row .get ('PathLevel',2 ),3 ))
            path_str ="/".join ([np .random .choice (phish_words )for _ in range (path_depth )])
            query_str =f"?user_id={idx }&token={''.join (np .random .choice (list (string .ascii_lowercase ),6 ))}"if row .get ('NumQueryComponents',0 )>0 else ""
            url =f"{protocol }{host }/{path_str }{query_str }"
        else :
            domain =np .random .choice (legit_domains )
            path_depth =int (min (row .get ('PathLevel',1 ),2 ))
            path_str ="/".join ([f"page{i +1 }"for i in range (path_depth )])if path_depth >0 else ""
            query_str =np .random .choice (legit_queries )if row .get ('NumQueryComponents',0 )>0 else ""
            url =f"{protocol }{domain }/{path_str }{query_str }"if path_str else f"{protocol }{domain }{query_str }"
        urls .append (url )
    return urls 
class CharLevelURLLSTM (nn .Module ):
    """
    PyTorch Recurrent Neural Network (LSTM) for Character-Level Raw URL Parsing.
    Processes URL strings token-by-token (character-by-character) through an Embedding + LSTM layer.
    """
    def __init__ (self ,vocab_size =VOCAB_SIZE ,embed_dim =32 ,hidden_dim =64 ,num_layers =2 ,dropout_rate =0.2 ):
        super (CharLevelURLLSTM ,self ).__init__ ()
        self .embedding =nn .Embedding (vocab_size ,embed_dim ,padding_idx =0 )
        self .lstm =nn .LSTM (
        input_size =embed_dim ,
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
    def forward (self ,char_token_ids ,seq_lengths ):
        embedded =self .embedding (char_token_ids )
        lstm_out ,(hn ,cn )=self .lstm (embedded )
        batch_size =char_token_ids .size (0 )
        idx =(seq_lengths -1 ).clamp (min =0 ).unsqueeze (1 ).unsqueeze (2 ).expand (batch_size ,1 ,lstm_out .size (2 ))
        last_char_out =lstm_out .gather (1 ,idx ).squeeze (1 )
        x =self .dropout (self .relu (self .fc1 (last_char_out )))
        out =self .sigmoid (self .fc2 (x ))
        return out 
def train_raw_url_char_lstm (epochs =20 ,batch_size =128 ,max_url_len =120 ,learning_rate =0.002 ):
    print ("Loading dataset...")
    df ,csv_path =load_dataset ()
    print ("\nGenerating character-tokenized URL text strings matching dataset properties...")
    url_strings =generate_urls_from_features (df )
    labels =df ['CLASS_LABEL'].values 
    print (f"Sample Phishing URL   (Class 0): {url_strings [0 ]}")
    print (f"Sample Legitimate URL (Class 1): {url_strings [5000 ]}")
    print (f"\nTokenizing URLs character-by-character into integer sequences (max_len={max_url_len })...")
    tokenized_data =[tokenize_url (url ,max_length =max_url_len )for url in url_strings ]
    X_tokenized =np .array ([t [0 ]for t in tokenized_data ])
    X_lengths =np .array ([t [1 ]for t in tokenized_data ])
    X_tr_tok ,X_te_tok ,len_tr ,len_te ,y_train ,y_test =train_test_split (
    X_tokenized ,X_lengths ,labels ,test_size =0.2 ,random_state =42 ,stratify =labels 
    )
    print (f"Train set: {X_tr_tok .shape [0 ]} URL sequences | Test set: {X_te_tok .shape [0 ]} URL sequences")
    X_tr_t =torch .tensor (X_tr_tok ,dtype =torch .long )
    len_tr_t =torch .tensor (len_tr ,dtype =torch .long )
    y_tr_t =torch .tensor (y_train ,dtype =torch .float32 ).unsqueeze (1 )
    X_te_t =torch .tensor (X_te_tok ,dtype =torch .long )
    len_te_t =torch .tensor (len_te ,dtype =torch .long )
    y_te_t =torch .tensor (y_test ,dtype =torch .float32 ).unsqueeze (1 )
    train_loader =DataLoader (TensorDataset (X_tr_t ,len_tr_t ,y_tr_t ),batch_size =batch_size ,shuffle =True )
    model =CharLevelURLLSTM (vocab_size =VOCAB_SIZE ,embed_dim =32 ,hidden_dim =64 ,num_layers =2 )
    criterion =nn .BCELoss ()
    optimizer =optim .Adam (model .parameters (),lr =learning_rate ,weight_decay =1e-4 )
    scheduler =optim .lr_scheduler .ReduceLROnPlateau (optimizer ,mode ='min',factor =0.5 ,patience =3 )
    print (f"\n--- Character-Level Tokenizer URL LSTM Architecture ---")
    print (model )
    print (f"\nTraining Character-Level LSTM (Parsing URL token-by-token) over {epochs } epochs...")
    train_losses ,val_losses =[],[]
    train_accs ,val_accs =[],[]
    for epoch in range (1 ,epochs +1 ):
        model .train ()
        running_tr_loss =0.0 
        correct_tr =0 
        total_tr =0 
        for batch_X ,batch_len ,batch_y in train_loader :
            optimizer .zero_grad ()
            out =model (batch_X ,batch_len )
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
            va_outputs =model (X_te_t ,len_te_t )
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
        final_probs =model (X_te_t ,len_te_t ).numpy ()
        final_preds =(final_probs >=0.5 ).astype (int )
    target_names =['Class 0: Malicious Phishing','Class 1: Legitimate']
    acc =accuracy_score (y_test ,final_preds )
    roc_auc =roc_auc_score (y_test ,final_probs )
    cm =confusion_matrix (y_test ,final_preds )
    report =classification_report (y_test ,final_preds ,target_names =target_names )
    print ("\n"+"="*60 )
    print ("      CHARACTER-LEVEL URL LSTM RESULTS")
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
    axes [0 ].set_title ('Character URL LSTM: Loss over Epochs',fontsize =12 ,fontweight ='bold')
    axes [0 ].set_xlabel ('Epoch',fontsize =10 )
    axes [0 ].set_ylabel ('Binary Cross-Entropy Loss',fontsize =10 )
    axes [0 ].legend (loc ='upper right')
    axes [0 ].grid (True ,linestyle =':',alpha =0.6 )
    axes [1 ].plot (range (1 ,epochs +1 ),train_accs ,label ='Train Accuracy',color ='#1f77b4',linewidth =2 )
    axes [1 ].plot (range (1 ,epochs +1 ),val_accs ,label ='Validation Accuracy',color ='#2ca02c',linewidth =2 ,linestyle ='--')
    axes [1 ].set_title ('Character URL LSTM: Accuracy over Epochs',fontsize =12 ,fontweight ='bold')
    axes [1 ].set_xlabel ('Epoch',fontsize =10 )
    axes [1 ].set_ylabel ('Accuracy (%)',fontsize =10 )
    axes [1 ].legend (loc ='lower right')
    axes [1 ].grid (True ,linestyle =':',alpha =0.6 )
    plt .tight_layout ()
    curve_plot_path =os .path .join (plots_dir ,"char_lstm_curves.png")
    plt .savefig (curve_plot_path ,dpi =300 )
    plt .close ()
    print (f"\nSaved training curves plot to      : {curve_plot_path }")
    plt .figure (figsize =(6 ,5 ))
    plt .imshow (cm ,interpolation ='nearest',cmap =plt .cm .Blues )
    plt .title ('Character URL LSTM: Confusion Matrix',fontsize =12 ,fontweight ='bold')
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
    cm_plot_path =os .path .join (plots_dir ,"char_lstm_confusion_matrix.png")
    plt .savefig (cm_plot_path ,dpi =300 )
    plt .close ()
    print (f"Saved confusion matrix plot to     : {cm_plot_path }")
    models_dir ="models"
    os .makedirs (models_dir ,exist_ok =True )
    model_path =os .path .join (models_dir ,"raw_url_char_lstm.pt")
    vocab_path =os .path .join (models_dir ,"char_vocab.pt")
    torch .save (model .state_dict (),model_path )
    joblib .dump (CHAR_TO_ID ,vocab_path )
    print (f"Saved Character LSTM weights to    : {model_path }")
    print (f"Saved Character Vocabulary to       : {vocab_path }")
    print ("="*60 )
    print ("\n--- Testing Real-Time Raw URL Parsing ---")
    test_urls =[
    "http://paypal-login-verify-account.top/signin.php?user=123",
    "https://google.com/search?q=phishing+detector",
    "https://wikipedia.org/wiki/Machine_learning"
    ]
    model .eval ()
    with torch .no_grad ():
        for test_url in test_urls :
            tok_ids ,seq_l =tokenize_url (test_url ,max_length =max_url_len )
            toks_t =torch .tensor ([tok_ids ],dtype =torch .long )
            lens_t =torch .tensor ([seq_l ],dtype =torch .long )
            prob =model (toks_t ,lens_t ).item ()
            pred_class ="Phishing (Class 0)"if prob <0.5 else "Legitimate (Class 1)"
            print (f"URL: {test_url :60s} -> Prob: {prob :.4f} | Prediction: {pred_class }")
    return model 
if __name__ =="__main__":
    train_raw_url_char_lstm (epochs =20 )
