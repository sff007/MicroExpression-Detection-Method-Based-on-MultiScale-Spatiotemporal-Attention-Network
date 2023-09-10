import numpy as np
import xlrd
import os
import random
# #CASME2
def get_label(sub,EP):
    data = xlrd.open_workbook("I:\\Mobile_v2\\dataset\\Dataset\\SAMM.xlsx")
    table = data.sheets()[0]
    rows = table.nrows
    cols = table.ncols

    for row in range(1, rows):
        for col in range(1, cols):
            if table.cell(row, 0).value == sub and table.cell(row,1).value == EP :
                emo = table.cell(row,11).value
                print('emo')
                labels = {'Contempt': '0', 'happiness': '1', 'surprise': '2', 'anger': '3', 'other': '4'} #SAMM
                lll = labels.get(emo)
                print('l',lll)

                break
            break
    return lll
def get_files(path,s,name_train,name_test):
    sub_path = []
    for  x in range(1,20):
        if(x<10):
            sub = 'sub0'+str(x)
            sub_path.append(sub)
        else:
            sub = 'sub' + str(x)
            sub_path.append(sub)
    print('ssssss', sub_path)

    for i in range(len(sub_path)):
        print('i',i)
        EP_path = os.listdir(path+'/'+sub_path[i])

        print('EP_path', EP_path)

        if (i+1 == s):

            for j in range(len(EP_path)):
                image_path = os.path.join(sub_path[i],EP_path[j])
                print('image',image_path)

                sub = sub_path[i][3:5]
                ep = EP_path[j]
                print('ep', ep)
                print('sub', sub)
                label1 = get_label(sub,ep)
                text_create(name_test, image_path, label1)
        else:
            for j in range(len(EP_path)):
                image_path = os.path.join(sub_path[i],EP_path[j])
                print('image',image_path)

                sub = sub_path[i][3:5]
                ep = EP_path[j]
                print('ep', ep)
                print('sub', sub)
                label1 = get_label(sub,ep)
                text_create(name_train,image_path,label1)


def text_create(name,image_path,label):
    desktop_path = "I:\\Mobile_v3\\dataset\\SAMM"
    full_path = desktop_path+'//'+name+'.txt'

    f = open(full_path,'a')
    f.write(image_path)
    f.write(' ')
    f.write(label)
    f.write('\n')
    f.close()
def shuffle(self):
    shuffle_file=open("./Shuffle_Data.txt",'w')
    temp=self.s
    np.random.shuffle(temp)
    for i in temp:
        shuffle_file.write(i[0]+" "+str(i[1])+"\n")
    return self.DataFile+"/Shuffle_Data.txt"
if __name__=='__main__':
    path = "I:\\Mobile_v3\\dataset\\Dataset\\SAMM"
    for s in range(27,28):
        name_train = 'train_'+str(s)
        name_test = 'test_'+str(s)
        get_files(path,s,name_train,name_test)