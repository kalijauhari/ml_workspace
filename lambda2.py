import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-08-05-16-36-52-899"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(endpoint_name=ENDPOINT, sagemaker_session=sagemaker.Session())

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image_data)

    # We return the data back to the Step Function    
    event['inferences'] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': {
            "info_data": json.dumps(event),
            "inferences": json.loads(event['inferences'])
        }
    }
