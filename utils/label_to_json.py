import json

label_class_file = '../data/label.txt'

label_class = list()

counter = 1
with open(label_class_file, 'r') as f:
    for cls_name in f:
        cls_name = cls_name.strip()
        label_class.append({"cls": counter,
                            "name":cls_name})
        counter += 1

json_data = json.dumps(label_class, indent=4)
print(json_data)

with open('../data/labels.json', 'w') as f:
    json.dump(label_class, f, indent=4)


