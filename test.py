import requests

resp = requests.post("http://localhost:5000/fgsm_untargeted",params={'eps':'0.1'}, files={"file": open('images/mastiff.jpg','rb')})
json = resp.json()
print(json['image'])