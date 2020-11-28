import boto3, requests, cv2, io, json
from PIL import Image
from pprint import pprint


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


session = boto3.Session(profile_name='default')
rekognition = session.client('rekognition')
# response = requests.get('https://upload.wikimedia.org/wikipedia/commons/thumb/8/88
# /Stephen_Hawking_David_Fleming_Martin_Curley.png/640px-Stephen_Hawking_David_Fleming_Martin_Curley.png')
img = Image.open('/Users/jorgeavila/Downloads/mask_sample.jpg')
response_content = image_to_byte_array(img)
rekognition_response = rekognition.detect_protective_equipment(
    Image={
        'Bytes': response_content
    },
    SummarizationAttributes={
        'MinConfidence': 0.85,
        'RequiredEquipmentTypes': [
            'FACE_COVER',
        ]
    })

# for obj in rekognition_response:
#     print("============================== Persons ==============================\n")
#     pprint (obj["Persons"])
#     print("=====================================================================\n")

json_object = json.dumps(rekognition_response, indent=4)
file = open("/Users/jorgeavila/Downloads/log.json", "w")
file.write(json_object)
file.close()

persons = rekognition_response["Persons"]
person_info = [b for b in persons]

person_bodypart = {
    "body part": [bp["BodyParts"] for bp in person_info]
}

for i in person_bodypart["body part"]:
    print('\n')
    for t in i:
        print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in t.items()) + "\n}")

# pprint(rekognition_response["Persons"][0]["BodyParts"][0]["EquipmentDetections"][0]["BoundingBox"]["Top"])
