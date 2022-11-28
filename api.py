from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods = ['GET'])
    

def interpreter():
    ax = request.args.get('ax', type=float)
    ay = request.args.get('ay', type=float)
    az = request.args.get('az', type=float)
    gx = request.args.get('gx', type=float)
    gy = request.args.get('gy', type=float)
    gz = request.args.get('gz', type=float)

    clf = joblib.load('SVC_model')
    acx, acy, acz, gyx, gyy, gyz = abs(float(ax)), abs(float(ay)), abs(float(az)), abs(float(gx)), abs(float(gy)), abs(float(gz))
    # test = np.array([acx,acy,acz,gyx,gyy,gyz]).reshape(-1,6)
    result = {}
    predicted = clf.predict([[ax, ay, az, gx, gy, gz]])
    print(predicted[0])
    result['output'] = str(predicted[0])
    return result

if __name__ == "__main__":
    app.run()