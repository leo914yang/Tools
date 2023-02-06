from glob import glob

lab_path = glob('/Users/leo/Downloads/labels/*')
img_path = glob('/Users/leo/Downloads/images/*')
lab = []
img = []
for i in lab_path:
    lab.append(''.join(i.split('labels/')[1].split('.png')))
print(lab)
for i in img_path:
    img.append(i.split('images/')[1])

# for i in img:
#     if i not in lab:
#         print(i)
