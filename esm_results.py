
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, accuracy_score, recall_score, precision_score, balanced_accuracy_score, precision_recall_curve
import numpy as np
from mlflow import log_metric, log_param, log_artifacts
import socket
np.set_printoptions(precision=3)

def F1_max_calc(y_true, y_proba1):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba1)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    return max_f1, max_f1_thresh

def ClassificationScore(run_name,seed_value, y_nox_test,y_proba_NOX, y_pdb_test,y_proba_PDB,data_read_time,training_time,threshold=0.5):
    print(training_time)
    roundplace=3
     
    ################### NOX DATA ###################
    dataset = "NOX "
    y_pred = (y_proba_NOX >= threshold).astype(int)       
    auc_score= roc_auc_score(y_nox_test, y_proba_NOX)   
    APS= average_precision_score(y_nox_test, y_proba_NOX) 
    F1_max,F1_max_thresh = F1_max_calc(y_nox_test, y_proba_NOX)     
    f1score = f1_score(y_nox_test, y_pred) 
    kappa=cohen_kappa_score(y_nox_test, y_pred)
    mcc= matthews_corrcoef(y_nox_test, y_pred)

    confusion = confusion_matrix(y_nox_test, y_pred)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]


    SP= TN/float(TN+FP)
    FPR = FP/float(TN+FP)
    FNR = FN/float(FN+TP)
    ACC = accuracy_score(y_nox_test, y_pred)
    PRAUC= average_precision_score(y_nox_test, y_proba_NOX)
    BACC= balanced_accuracy_score(y_nox_test, y_pred)
    Recall = recall_score(y_nox_test, y_pred) 
    Precision = precision_score(y_nox_test, y_pred)
    TNR = TN/float(TN+FP)
    TPR = TP/float(TP+FN)
    
     # ######CAID2 NOX######
    SOTA_NOX_AUC=0.885
    SOTA_NOX_APS=0.754
    SOTA_NOX_F1_max=0.750
    
    
    print("\n","*"*20," NOX Improvement","*"*20, "\n")  
    
    auc_score = round(auc_score,roundplace) 
    APS = round(APS,roundplace)
    F1_max = round(F1_max,roundplace)

    Imp_NOX_AUC = ((auc_score-SOTA_NOX_AUC)/SOTA_NOX_AUC)*100
    Imp_NOX_APS = ((APS-SOTA_NOX_APS)/SOTA_NOX_APS)*100
    Imp_NOX_F1_max = ((F1_max-SOTA_NOX_F1_max)/SOTA_NOX_F1_max)*100
    Imp_NOX_Total=(Imp_NOX_AUC+Imp_NOX_APS+Imp_NOX_F1_max)/3
    
    Imp_NOX_AUC = round(Imp_NOX_AUC,roundplace) 
    Imp_NOX_APS = round(Imp_NOX_APS,roundplace)
    Imp_NOX_F1_max = round(Imp_NOX_F1_max,roundplace)
    Imp_NOX_Total = round(Imp_NOX_Total,roundplace)

    res=open("./Results/"+run_name+"_results.txt","a")
    print(" Imp NOX (%) AUC ,",     round(Imp_NOX_AUC,roundplace)       , end =" ")
    print(" Imp NOX (%) F1_max,",  round(Imp_NOX_F1_max,roundplace)      , end =" ")
    print(" Imp NOX (%) APS,",   round(Imp_NOX_APS,roundplace)         , end =" " )
    print(" ImpTotal NOX ",   round(Imp_NOX_Total,roundplace)   , end =" ")

    print(" Imp NOX (%) AUC ,",     round(Imp_NOX_AUC,roundplace)       , end =" ",file=res)
    print(" Imp NOX (%) F1_max,",  round(Imp_NOX_F1_max,roundplace)      , end =" ",file=res)
    print(" Imp NOX (%) APS,",   round(Imp_NOX_APS,roundplace)         , end =" " ,file=res)
    print(" ImpTotal NOX ",   round(Imp_NOX_Total,roundplace)  ,file=res)
    # log metrics in mlflow
    
    log_metric("0_"+dataset+"Imp AUC", Imp_NOX_AUC)  
    log_metric("0_"+dataset+"Imp F1_max", Imp_NOX_F1_max)
    log_metric("0_"+dataset+"Imp APS", Imp_NOX_APS)
    log_metric("0_"+dataset+"ImpTotal ", Imp_NOX_Total)
    
      
    
    
    log_metric("1_"+dataset+"AUC", auc_score)  
    log_metric("1_"+dataset+"F1_max", F1_max)
    log_metric("1_"+dataset+"APS", APS)
    log_metric("1_"+dataset+"F1_max_thresh", F1_max_thresh) 
    
    
    
    log_metric(dataset+"F1-score", f1score)
    log_metric(dataset+"MCC", mcc)
    log_metric(dataset+"Kappa", kappa)   
    
 
    log_metric(dataset+"TP", TP)
    log_metric(dataset+"TN", TN)
    log_metric(dataset+"FP", FP)
    log_metric(dataset+"FN", FN)     
    
    log_metric(dataset+"SP", SP)
    log_metric(dataset+"FPR", FPR)
    log_metric(dataset+"FNR", FNR)
    log_metric(dataset+"ACC", ACC)
    log_metric(dataset+"PRAUC", PRAUC)
    log_metric(dataset+"BACC", BACC)
    log_metric(dataset+"Recall", Recall)
    log_metric(dataset+"Precision", Precision)
    log_metric(dataset+"Seed", seed_value)
    log_metric(dataset+"TNR", TNR)
    log_metric(dataset+"TPR", TPR)
    log_param(dataset+"HostName", socket.gethostname())
    
    log_metric("1_"+dataset+"threshold", threshold)
    log_metric("1_"+dataset+"training_time", training_time)
    log_metric("1_"+dataset+"dataread_time", data_read_time)        
    


    ################### PDB DATA ###################
    dataset = "PDB "    
    y_pred = (y_proba_PDB >= threshold).astype(int)       
    auc_score= roc_auc_score(y_pdb_test, y_proba_PDB)  
    APS= average_precision_score(y_pdb_test, y_proba_PDB) 
    F1_max,F1_max_thresh = F1_max_calc(y_pdb_test, y_proba_PDB)     
    f1score = f1_score(y_pdb_test, y_pred) 
    kappa=cohen_kappa_score(y_pdb_test, y_pred)
    mcc= matthews_corrcoef(y_pdb_test, y_pred)

    confusion = confusion_matrix(y_pdb_test, y_pred)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]


    SP= TN/float(TN+FP)
    FPR = FP/float(TN+FP)
    FNR = FN/float(FN+TP)
    ACC = accuracy_score(y_pdb_test, y_pred)
    PRAUC= average_precision_score(y_pdb_test, y_proba_PDB)
    BACC= balanced_accuracy_score(y_pdb_test, y_pred)
    Recall = recall_score(y_pdb_test, y_pred) 
    Precision = precision_score(y_pdb_test, y_pred)
    TNR = TN/float(TN+FP)
    TPR = TP/float(TP+FN)
    
     # ######CAID2######
    SOTA_PDB_AUC=0.955
    SOTA_PDB_APS=0.928
    SOTA_PDB_F1_max=0.859
    
    
    
    print("\n","*"*20," PDB Improvement","*"*20, "\n")  
    
    auc_score = round(auc_score,roundplace) 
    APS = round(APS,roundplace)
    F1_max = round(F1_max,roundplace)
    

    Imp_PDB_AUC = ((auc_score-SOTA_PDB_AUC)/SOTA_PDB_AUC)*100
    Imp_PDB_APS = ((APS-SOTA_PDB_APS)/SOTA_PDB_APS)*100
    Imp_PDB_F1_max = ((F1_max-SOTA_PDB_F1_max)/SOTA_PDB_F1_max)*100
    Imp_PDB_Total=(Imp_PDB_AUC+Imp_PDB_APS+Imp_PDB_F1_max)/3
    
    Imp_PDB_AUC = round(Imp_PDB_AUC,roundplace) 
    Imp_PDB_APS = round(Imp_PDB_APS,roundplace)
    Imp_PDB_F1_max = round(Imp_PDB_F1_max,roundplace)
    Imp_PDB_Total = round(Imp_PDB_Total,roundplace)    


   
    print(" Imp PDB (%) AUC ,",     round(Imp_PDB_AUC,roundplace)       , end =" ")
    print(" Imp PDB (%) F1_max,",  round(Imp_PDB_F1_max,roundplace)      , end =" ")
    print(" Imp PDB (%) APS,",   round(Imp_PDB_APS,roundplace)         , end =" " )
    print(" ImpTotal PDB ",   round(Imp_PDB_Total,roundplace)   , end =" ")

    
    print(" Imp PDB (%) AUC ,",     round(Imp_PDB_AUC,roundplace)       , end =" ",file=res)
    print(" Imp PDB (%) F1_max,",  round(Imp_PDB_F1_max,roundplace)      , end =" ",file=res)
    print(" Imp PDB (%) APS,",   round(Imp_PDB_APS,roundplace)         , end =" ",file=res) 
    print(" ImpTotal PDB ",   round(Imp_PDB_Total,roundplace)   , end =" ",file=res)
    
    # log metrics in mlflow
    
    log_metric("0_"+dataset+"Imp AUC", Imp_PDB_AUC)  
    log_metric("0_"+dataset+"Imp F1_max", Imp_PDB_F1_max)
    log_metric("0_"+dataset+"Imp APS", Imp_PDB_APS)
    log_metric("0_"+dataset+"ImpTotal ", Imp_PDB_Total)   

    log_metric("1_"+dataset+"AUC", auc_score)  
    log_metric("1_"+dataset+"F1_max", F1_max)
    log_metric("1_"+dataset+"APS", APS)
    log_metric("1_"+dataset+"F1_max_thresh", F1_max_thresh) 
     
    log_metric(dataset+"F1-score", f1score)
    log_metric(dataset+"MCC", mcc)
    log_metric(dataset+"Kappa", kappa) 
 
    log_metric(dataset+"TP", TP)
    log_metric(dataset+"TN", TN)
    log_metric(dataset+"FP", FP)
    log_metric(dataset+"FN", FN)     
    
    log_metric(dataset+"SP", SP)
    log_metric(dataset+"FPR", FPR)
    log_metric(dataset+"FNR", FNR)
    log_metric(dataset+"ACC", ACC)
    log_metric(dataset+"PRAUC", PRAUC)
    log_metric(dataset+"BACC", BACC)
    log_metric(dataset+"Recall", Recall)
    log_metric(dataset+"Precision", Precision)
    log_metric(dataset+"Seed", seed_value)
    log_metric(dataset+"TNR", TNR)
    log_metric(dataset+"TPR", TPR)
    log_param(dataset+"HostName", socket.gethostname())
    
    log_metric("1_"+dataset+"threshold", threshold)
    log_metric("1_"+dataset+"training_time", training_time)
    log_metric("1_"+dataset+"dataread_time", data_read_time)      
    
    

    stats(y_nox_test, y_pdb_test)


# print dataset statistics
def stats( y_nox_test, y_pdb_test ):
    y_nox_test_1=sum(y_nox_test==1)
    y_nox_test_0=sum(y_nox_test==0)
    y_pdb_test_1=sum(y_pdb_test==1)
    y_pdb_test_0=sum(y_pdb_test==0)

    # log metrics in mlflow

    log_metric("y_nox_test_1", y_nox_test_1)
    log_metric("y_nox_test_0", y_nox_test_0)
    log_metric("y_pdb_test_1", y_pdb_test_1)
    log_metric("y_pdb_test_0", y_pdb_test_0)
    







    

    
    
