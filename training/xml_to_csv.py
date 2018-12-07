import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def xml_to_csv(pathlist):
    xml_list = []
    #for xml_file in glob.glob(path + '/*_label/*.xml'):
    for xml_file in pathlist:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = root.find('path').text
        path = path.replace('/jixj/term3/p015', os.getcwd())
        path = path.replace('/home/iquantela/Study/CarND-Capstone/training', os.getcwd())
        for member in root.findall('object'):
            value = (path,
                     #root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(argv):
    #print (argv[1])
    image_path = os.path.join(os.getcwd(), argv[1])
    #image_path = os.getcwd()
    image_list = shuffle(glob.glob(image_path + '/*_label/*.xml')) 
    train, test = train_test_split(image_list, test_size=0.2, random_state=0)    

    xml_df = xml_to_csv(train)
    output = './train/' + argv[1] + '_train.csv'
    xml_df.to_csv(output, index=None)
    print('Successfully converted xml to csv. \noutput:  ' +  output )

    xml_df = xml_to_csv(test)
    output = './test/' + argv[1] + '_test.csv'
    xml_df.to_csv(output, index=None)
    print('Successfully converted xml to csv. \noutput:  ' +  output )




if __name__ == '__main__':
    main(sys.argv)
