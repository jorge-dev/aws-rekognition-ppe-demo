from chalice import Chalice
import boto3

app = Chalice(app_name='rekogDemo')


@app.route('/')
def index():
    return {'hello': 'world'}

@app.on_s3_event(bucket='image-rekognition-s3')
def handler(event):
    print("Object uploaded for bucket: %s, key: %s" %(event.bucket, event.key))

    #recognition code

    filename = event.key
    bucketName = event.bucket
    client = boto3.client('rekognition')
    response = client.detect_labels(Image={
        'S3Object': {
            'Bucket': bucketName,
            'Name': filename,
        },
    },)
     
    for label in response['Labels']:
        print(label['Name'] + ":" + str(label['Confidence']))





# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
