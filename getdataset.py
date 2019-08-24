import json
import requests

def imgdownload():
        count = 0

        for line in open('Indian_Number_plates.json', 'r'):
                json_data = json.loads(line)
                count += 1
                image_url = json_data['content']
                annotation = json_data['annotation']
                img_data = requests.get(image_url).content
                with open('images/'+str(count)+'.jpg', 'wb') as handler:
                        handler.write(img_data)
def getcoords():
        count=0
        for line in open('Indian_Number_plates.json', 'r'):
                json_data = json.loads(line)
                count += 1
                #image_url = json_data['content']
                annotation = json_data['annotation']
                coord1=annotation[0]['points'][0]['x']
                coord2=annotation[0]['points'][0]['y']
                coord3=annotation[0]['points'][1]['x']
                coord4=annotation[0]['points'][1]['y']
                with open('images/'+str(count)+'.txt', 'w') as handler:
                        handler.write(str(coord1)+' '+str(coord2)+' '+str(coord3)+' '+str(coord4))


getcoords()
# with open('Indian_Number_plates.json',encoding='utf-8-sig') as json_file:
#     json_data = json.load(json_file)
#     for img in range(len(json_data)):
#         count += 1
#         url = img['content']
#         annotation = img['annotation']
#         img_data = requests.get(image_url).content
#         with open(str(count)+'.jpg', 'wb') as handler:
#             handler.write(images/img_data)
