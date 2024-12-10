import pickle

from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    # unpickle header and forest in forest.p
    infile = open("forest.p", "rb")
    header, forest = pickle.load(infile)
    infile.close()
    return header, forest
        
# we need to add some routes!
# a "route" is a function that handles a request
# e.g. for the HTML content for a home page
# or for the JSON response for a /predict API endpoint, etc
@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the stress predictor app</h1>", 200
        

# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    screen_on_time = request.args.get("screen_on_time") # defaults to None
    num_sms = request.args.get("num_sms")
    num_calls = request.args.get("num_calls")
    sleep_time = request.args.get("sleep_time")
    instance = [[screen_on_time, num_sms, num_calls, sleep_time]]
    header, forest = load_model()
    # lets make a prediction!
    pred = forest.predict(instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400


if __name__ == "__main__":
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host="0.0.0.0", port=5001, debug=False)
    # TODO: when deploy app to "production", set debug=False
    # and check host and port values

    # instructions for deploying flask app to render.com: https://docs.render.com/deploy-flask