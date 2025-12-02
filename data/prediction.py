import joblib

def predicr(data):
    clf = joblib.load("knn_model.sav")
    return clf.predict(data)