from sklearn.metrics import confusion_matrix

def score(y_true,y_pred):
    arr = confusion_matrix(y_true,y_pred)
    pd = arr[1][1]/(arr[1][1]+arr[1][0])
    pf = arr[0][1]/(arr[0][0]+arr[0][1])
    fallout = 1-pf
    g_measure = 2*((pd*fallout)/(pd+fallout))
    return g_measure
