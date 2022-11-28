import pandas as pd
import pytesseract
from pytesseract import Output
import cv2
import json
from pdf2image import convert_from_path
import numpy as np
import os
import shutil

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.2.0/share/tessdata'
multipliersDict = json.load(open('multipliersDictNew.json'))
dataFrameResults = pd.DataFrame()

# print("\n\nINICIANDO A CONVERSÃO DE PDF\n\n")

# #Converting PDF's to PNG
# for file in os.listdir('PDFs'):
#   if file.split('.')[-1]=='pdf':
#     print(file)
#     pdfFile = convert_from_path(f'PDFs/{file}',dpi=300)
    
#     for i in range(len(pdfFile)):
#         # Save pages as images in the pdf
#         pdfFile[i].save(f'PNGs/{file.split(".")[0]}_page{i}.png', 'PNG')


# 
# 
# 
# 

print("\n\nINICIANDO A LEITURA DE IMAGENS")

whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÇÁÉÍÓÚÃÕÂÊÎÔÛ-/'
blacklist = 'abcdefghijklmnopqrstuvwxyzçáéíóúãõâêîôû*&|º?[]\}{+=˚'

for file in os.listdir('PNGs'):
  if file.split('.')[-1]=='png':

        img = cv2.imread(f'PNGs/{file}') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
        results = pytesseract.image_to_data(img, output_type=Output.DICT,lang='por',config=f'-c tessedit_char_blacklist={blacklist}')

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


        #generating DF with the results from OCR detection and calculating new columns with the space between words
        df = pd.DataFrame.from_dict(results)
        df['right'] = df['left']+df['width']
        df['space'] = np.nan
        df['key'] = np.nan
        df['spaceRelativeToHeight'] = np.nan

        keyList = []
        spaceRelativeToHeightList = []

        for i in range(len(df)):
            keyList.append(str(df['page_num'][i])+str(df['block_num'][i])+str(df['par_num'][i])+str(df['line_num'][i]))
            try:
                diff = (df['left'][i]-df['right'][i-1])/df['height'][i-1]
                # if diff<0:
                #     diff=0
            except:
                pass
                diff = np.nan

            spaceRelativeToHeightList.append(diff)
            
        df['containsLower'] = df['text'].str.match(r'(?=.*[a-z])')
        df['key'] = keyList
        df['spaceRelativeToHeight'] = spaceRelativeToHeightList






        #Concatenating nearby words into terms/phrases using new columns calculated

        dfWords = df[(df['level']==5)&(df['containsLower']==False)&(df['text']!=' ')].reset_index().drop(columns=['index'])

        phrases={}
        phrase=''
        currentKey = dfWords['key'][0]
        left = dfWords['left'][0]
        top = dfWords['top'][0]
        width = dfWords['width'][0]
        height = dfWords['height'][0]

        width=0


        for i in range(len(dfWords)):
            if (dfWords['spaceRelativeToHeight'][i]<=1) and (dfWords['key'][i]==currentKey):
                # print(dfWords['text'][i])
                phrase+=(dfWords['text'][i]+' ')
                width+=dfWords['width'][i]
            else:
                # print(phrase)
                phrases[phrase.strip()] = {'left':left, 'top':top, 'width':width, 'height':height}

                currentKey = dfWords['key'][i]
                left= dfWords['left'][i]
                top= dfWords['top'][i]
                height= dfWords['height'][i]
                width= dfWords['width'][i]
                phrase = (dfWords['text'][i]+' ')


        # print(phrases)


        #Plotting the rectangles over identified terms, using multiplier values 
        #Cropping original images into smaller ones containing fields

        whitelistCrop = '0123456789.'
        blacklistCrop = 'abcdefghijklmnopqrstuvwxyzçáéíóúãõâêîôû&º?[]}{+=˚|!$'

        placaPlaceholder = file
        img2=img

        try:
            os.mkdir(f'Fields/{placaPlaceholder}')
        except:
            pass

        
        dictDataFrame = {}

        for i in phrases.keys():
            if i in multipliersDict.keys():
                # print(i)
                x = int(phrases[i]['left'])-10
                y = int(phrases[i]['top'] + phrases[i]['height'])
                xw = int(phrases[i]['left'] + multipliersDict[i]['width']*(phrases[i]['width']))+10
                yh = int(phrases[i]['top'] + phrases[i]['height'] + multipliersDict[i]['height'])
                
                crop_img = img[y:yh, x:xw]
                if i == "EIXOS":
                    resultsCrop = pytesseract.image_to_data(crop_img, output_type=Output.DICT,lang='por',config=f'--psm 10  --oem 3 -c tessedit_char_whitelist=0123456789')
                else:
                    resultsCrop = pytesseract.image_to_data(crop_img, output_type=Output.DICT,lang='por',config=f'--psm 11 --oem 3 -c tessedit_char_blacklist={blacklistCrop}')
                
                fieldNameWords = i.split(' ')
                cropWords = resultsCrop['text']
                fieldAnswer = ' '.join(cropWords)
                
                if i == 'PLACA':
                    placaReal = fieldAnswer.strip()
                    print('Placa real', placaReal)
                dictDataFrame[str(i).replace('/','-')] = [fieldAnswer.strip()]
               

                # print(i,'------>',fieldAnswer)
                cv2.imwrite(f"Fields/{placaPlaceholder}/{i.replace('/','-')}.png", crop_img)
                cv2.rectangle(img2, (x,y), (xw,yh), (0, 0, 255), 7)
        try:
            shutil.rmtree(f"Fields/{placaReal}")
        except:
            pass
        os.rename(f"Fields/{placaPlaceholder}",f"Fields/{placaReal}")
        cv2.imwrite(f"Boxes/{placaReal}.png", img2)
        dataFrameResults = pd.concat([dataFrameResults, pd.DataFrame.from_dict(dictDataFrame,orient='columns')], ignore_index = True)
    
dataFrameResults.to_csv('dfCompiled.csv')
