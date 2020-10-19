from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from stacktags import model_utils
import pickle
import os
import uvicorn

app = FastAPI()

model = load_model(os.path.join('models', 'feedforward_model.h5'))


class StackPost(BaseModel):
    title: str
    body: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/tagspredict")
async def root():
    return {"message": "Predict stack exchange tags!"}


@app.post('/tagspredict')
async def predict_tags(post: StackPost):

    input_data = model_utils.preprocess_input(post.title, post.body)
    predictions = model.predict(input_data)[0]

    with open('count_vectorizer.pck', 'rb') as handle:
        countvectorizer = pickle.load(handle)

    features = countvectorizer.get_feature_names()

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    tags_dict = {k: v for k, v in zip(features, predictions) if v == 1}

    return {
        "output": list(tags_dict.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
