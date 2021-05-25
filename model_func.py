
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from io import StringIO
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from IPython.display import display, display_html 


# split our X and y
# do the capital X, lowercase y thing for train test and split
# X is the data frame of the features, y is a series of the target
def split_Xy (df, column):
    '''
    Take in a DataFrame (train, validate, test) and return X and y; .
    df: train, validate or  test. Select one
    column: which columns you want to  stratify on. Ex. stratify on 'survived'
    return X, y  DataFrames.
    Example:
    X_validate, y_validate = model.split_Xy(validate, 'survived') 
    '''
    X_df = df.drop(columns= column)
    y_df = df[[column]]
    return X_df, y_df

def model_performs (X_df, y_df, model):
    '''
    Take in a X_df, y_df and model  and fit the model , make a prediction, calculate score (accuracy), 
    confusion matrix, rates, clasification report.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model: name of your model that you prevously created 
    
    Example:
    mmodel_performs (X_train, y_train, model1)
    '''

    #prediction
    pred = model.predict(X_df)

    #score = accuracy
    acc = model.score(X_df, y_df)

    #conf Matrix
    conf = confusion_matrix(y_df, pred)
    mat =  pd.DataFrame ((confusion_matrix(y_df, pred )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat.columns, index=mat.index)
    cf = rubric_df + ': ' + mat.values.astype(str)

    #assign the values
    tp = conf[1,1]
    fp =conf[0,1] 
    fn= conf[1,0]
    tn =conf[0,0]

    #calculate the rate
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = tn/(tn+fp)
    fnr = fn/(fn+tp)

    #classification report
    clas_rep =pd.DataFrame(classification_report(y_df, pred, output_dict=True)).T
    clas_rep.rename(index={'0': "dead", '1': "survived"}, inplace = True)
    print(f'''
    The accuracy for our model is {acc:.4%}

    The True Positive Rate is {tpr:.3%},    The False Positive Rate is {fpr:.3%},
    The True Negative Rate is {tnr:.3%},    The False Negative Rate is {fnr:.3%}

    ________________________________________________________________________________
    ''')
    print('''
    The positive is  'survived'

    Confusion Matrix
    ''')
    display(cf)
    print('''

    ________________________________________________________________________________
    
    Classification Report:
    ''')
    display(clas_rep)
   




def dec_tree(model, X_df):
    '''
    Plot a decision tree.
    Take in a model, X_df  
    model: name of your model that you prevously created 
    X_df: train, validate or  test. Select one
    
    Example:
    model.dec_tree(model1, X_train)
    '''
    plt.figure(figsize=(24, 12))
    plot_tree(
    model,
    feature_names=X_df.columns.tolist(),
    class_names=['died', 'survived'],
    )
    plt.show()



def compare (model1, model2, X_df,y_df):
    '''
    Take in a X_df, y_df and model  and fit the model , make a prediction, calculate score (accuracy), 
    confusion matrix, rates, clasification report.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model: name of your model that you prevously created 
    
    Example:
    mmodel_performs (X_train, y_train, model1)
    '''



    #prediction
    pred1 = model1.predict(X_df)
    pred2 = model2.predict(X_df)

    #score = accuracy
    acc1 = model1.score(X_df, y_df)
    acc2 = model2.score(X_df, y_df)


    #conf Matrix
    #model 1
    conf1 = confusion_matrix(y_df, pred1)
    mat1 =  pd.DataFrame ((confusion_matrix(y_df, pred1 )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat1.columns, index=mat1.index)
    cf1 = rubric_df + ': ' + mat1.values.astype(str)
    
    #model2
    conf2 = confusion_matrix(y_df, pred2)
    mat2 =  pd.DataFrame ((confusion_matrix(y_df, pred2 )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    cf2 = rubric_df + ': ' + mat2.values.astype(str)
    #model 1
    #assign the values
    tp = conf1[1,1]
    fp =conf1[0,1] 
    fn= conf1[1,0]
    tn =conf1[0,0]

    #calculate the rate
    tpr1 = tp/(tp+fn)
    fpr1 = fp/(fp+tn)
    tnr1 = tn/(tn+fp)
    fnr1 = fn/(fn+tp)

    #model 2
    #assign the values
    tp = conf2[1,1]
    fp =conf2[0,1] 
    fn= conf2[1,0]
    tn =conf2[0,0]

    #calculate the rate
    tpr2 = tp/(tp+fn)
    fpr2 = fp/(fp+tn)
    tnr2 = tn/(tn+fp)
    fnr2 = fn/(fn+tp)

    #classification report
    #model1
    clas_rep1 =pd.DataFrame(classification_report(y_df, pred1, output_dict=True)).T
    clas_rep1.rename(index={'0': "dead", '1': "survived"}, inplace = True)

    #model2
    clas_rep2 =pd.DataFrame(classification_report(y_df, pred2, output_dict=True)).T
    clas_rep2.rename(index={'0': "dead", '1': "survived"}, inplace = True)
    print(f'''
    ******       Model 1  ******                                ******     Model 2  ****** 
    The accuracy for our model 1 is {acc1:.4%}            |   The accuracy for our model 2 is {acc2:.4%}  
                                                        |
    The True Positive Rate is {tpr1:.3%}                   |   The True Positive Rate is {tpr2:.3%}  
    The False Positive Rate is {fpr1:.3%}                   |  The False Positive Rate is {fpr2:.3%} 
    The True Negative Rate is {tnr1:.3%}                   |   The True Negative Rate is {tnr2:.3%} 
    The False Negative Rate is {fnr1:.3%}                   |  The False Negative Rate is {fnr2:.3%}

    _____________________________________________________________________________________________________________
    ''')
    print('''
    The positive is  'survived'

    Confusion Matrix
    ''')
    cf1_styler = cf1.style.set_table_attributes("style='display:inline'").set_caption('Confusion Matrix')
    cf2_styler = cf2.style.set_table_attributes("style='display:inline'").set_caption('Confusion Matrix')
    space = "\xa0" * 10
    display_html(cf1_styler._repr_html_()+ space  + cf2_styler._repr_html_(), raw=True)
    # print(display(cf1),"           ", display(cf2))
    
    print('''

    ________________________________________________________________________________
    
    Classification Report:
    ''')
    display(clas_rep1), display(clas_rep2)
   

# print(f'''
# positive: {positive}

#          | accuracy | recall | precision
#          | -------- | ------ | ---------         
#    model | {model_accuracy:8.1%} | {model_recall:6.1%} | {model_precision:9.1%}
# baseline | {baseline_accuracy:8.1%} | {baseline_recall:6.1%} | {baseline_precision:9.1%}
# '''