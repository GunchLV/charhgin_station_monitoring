import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# selenium 4
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.remote import webelement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from bs4 import BeautifulSoup
import os
from datetime import datetime
from selenium.webdriver import ActionChains
from paroles import vards, parole

def laiks_tagad():
    laiks = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    return laiks

kaut_kas_atrasts = False

attēla_nr = 2
bilde = ['car_img1.jpg','car_img2.jpg','no_stamerienas.jpg','arzemju_nr.jpg','face.jpg'][attēla_nr-1]
img= cv2.imread(bilde)

print('izejas dati')

plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plot.axis('off')
plot.show()

kaut_kas_atrasts = False

# aizmiglojam seju, ja tāda ir
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
harr_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cords = harr_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=1)
for x, y, w, h in face_cords:
    blur_face = img[y:y+h, x:x+w]
    blur_face = cv2.GaussianBlur(blur_face,(23, 23), 30)
    img[y:y+blur_face.shape[0], x:x+blur_face.shape[1]] = blur_face
    
reader = easyocr.Reader(['en'])
RST = reader.readtext(img)

if RST==[]:
    print('Attēlā teksts vai skaitļi nav atrasti')
    auto_numurs=''
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.axis('off')
    plot.show()
else:          
    print('Atrasts teksts vai skaitļi')
    kaut_kas_atrasts=True
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.axis('off')
    plot.show()
    print(RST)


atrastie_teksti = []
for variants in RST:
    atrastie_teksti.append(variants[1])
    
parbaude = pd.DataFrame(columns=['varianti'], data=atrastie_teksti)
parbaude['ir_domu_zime'] = np.where(parbaude['varianti'].astype(str).str.contains('-'),1,0)
parbaude['garāks_par_3'] = np.where(parbaude['varianti'].astype(str).str.len()>3,1,0)
parbaude['īsāks_par_11'] = np.where(parbaude['varianti'].astype(str).str.len()<11,1,0)
parbaude['vertejums_kopa'] = parbaude.iloc[:,1:].sum(axis=1)
parbaude=parbaude.sort_values(by='vertejums_kopa', ascending=False)
parbaude = parbaude.reset_index(drop=True)
print('iespējamākais auto numurs --> ' + parbaude['varianti'][0])
parbaude

if kaut_kas_atrasts==True:
    #pafiltrējam atrastos rezultātus un atstājam, tos, kas visticamāk ir auto nr.
    pareizais = RST[atrastie_teksti.index(parbaude['varianti'][0])]
    
    jaunais=[]
    for pirmais in pareizais[0]:
        for otrais in pirmais:
            jaunais.append(int(otrais))
            
    jaunais = [[jaunais[0], jaunais[1]], [jaunais[2], jaunais[3]], [jaunais[4], jaunais[5]], [jaunais[6], jaunais[7]]]
    pareizais = (jaunais, pareizais[1], pareizais[2])

    augsa_kreisais = tuple(pareizais[0][0])
    leja_labais = tuple(pareizais[0][2])
    auto_numurs = pareizais[1]
    auto_numurs = auto_numurs.replace(':','')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    img = cv2.rectangle(img,augsa_kreisais,leja_labais,(4, 16, 189),3)
    #img = cv2.putText     # šis uzrakstītu uz bildes tekstu
    print('atrastais auto numurs --> ' + auto_numurs + '\n')
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot.axis('off')
    plot.show()
    
    # pēc numura izmēra vēl vajadzētu izrēķināt attālumu!
if auto_numurs!='':
    numura_vidus_horizontali = (pareizais[0][2][0] - pareizais[0][1][0])/2 + pareizais[0][1][0]
    bildes_platums = img.shape[1]
    bildes_vidus_horizontali = bildes_platums/2
    nobidijums = round((numura_vidus_horizontali - bildes_vidus_horizontali)/bildes_platums*100, 2)
    novietojums = 'normāls' if abs(nobidijums)<15 else 'pāri līnijām'
    if nobidijums > 0:
        print('auto novietots ' + str(nobidijums) + '% no bildes platuma pa labi. Novietojums - ' + novietojums)
    else:
        print('auto novietots ' + str(abs(nobidijums)) + '% no bildes platuma pa kreisi. Novietojums - ' + novietojums)
        
        #salīdzina vai atrastais nr ir tas, kas tur jau stāv?
vecie_dati = pd.read_csv('datu_tabula.csv')
if auto_numurs=='':
    auto_pie_stacijas=False
else:
    if vecie_dati['auto_nr'].to_list()[-1] != auto_numurs:  # ja pēdējā ieraksta auto numurs nav šobrīd fiksētais
        # ja 1. reize palaižot svaigu skriptu vai beidzies login laiks
        try:
            driver.get('https://e.csdd.lv/tadati/')
            inputElement = driver.find_element(by=By.ID, value= 'rn')
        except:
            driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))
            driver.get('https://e.csdd.lv/tadati/')
            ieejas_poga = driver.find_element(by=By.ID, value= 'goLogin')
            ieejas_poga.click()
            
            epasta_vieta = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "email")))
            epasta_vieta = driver.find_element(by=By.ID, value= 'email')
            epasta_vieta.send_keys(vards)
            paroles_vieta = driver.find_element(by=By.ID, value= 'psw')
            paroles_vieta.send_keys(parole)
            paroles_vieta.send_keys(Keys.ENTER)
        
        driver.get('https://e.csdd.lv/tadati/')
        inputElement = driver.find_element(by=By.ID, value= 'rn')
        inputElement.send_keys(auto_numurs)
        #inputElement.send_keys('neīsts numurs testam')
        inputElement.send_keys(Keys.ENTER)
        
        try:
            elem = driver.find_element(by=By.ID, value='refer-table')
            soup = BeautifulSoup(elem.get_attribute('innerHTML'), 'html.parser')
            lapas_teksts = soup.find_all('td')
            
            salabots = str(lapas_teksts).replace('\n\t\t\t','')
            salabots = salabots.replace('</td>,','')
            salabots = salabots.split('<td>')
            salabots = [salab.strip() for salab in salabots]
            
            degv_veids = ''
            marka_mod = ''
            for i in range(0, len(salabots)):
                if 'Degvielas veids' in salabots[i]:
                    degv_veids = salabots[i+1]
                if 'modelis' in salabots[i]:
                    marka_mod = salabots[i+1]
            
            auto_pie_stacijas=True
            #print('Auto marka/modelis ir ' + marka_mod + ' un degvielas veids ir ' + degv_veids)  
            
        except:
            auto_pie_stacijas=False
            #print('šobrīd auto nav redzams')
    else:
        auto_pie_stacijas=True
        
        vecie_dati = pd.read_csv('datu_tabula.csv')
if auto_pie_stacijas==True:
    if vecie_dati['auto_nr'].to_list()[-1] ==auto_numurs:  # ja pēdējā ieraksta auto numurs == šobrīd fiksētais
        jauns_ieraksts = vecie_dati.tail(1).copy()
        jauns_ieraksts['aizbrauksanas_laiks'] = pd.NA
        jauns_ieraksts['statuss'] = 'stāv'
        jauns_ieraksts['timestamp'] = laiks_tagad()
        dati = pd.concat([vecie_dati,jauns_ieraksts], ignore_index=True)
    else:
        jauns_ieraksts = pd.DataFrame(data={'auto_nr':auto_numurs, 'marka_modelis':marka_mod,
                                            'degvielas_veids':degv_veids, 'auto_novietojums':novietojums,'uzlades_stacijas_nr':1234, 'ierasanas_laiks':laiks_tagad(),
                                            'aizbrauksanas_laiks':pd.NA, 'statuss':'ieradies', 'timestamp':laiks_tagad()}, index=[0])
        dati = pd.concat([vecie_dati,jauns_ieraksts], ignore_index=True)
else:
    if (auto_numurs=='') & (vecie_dati['statuss'].to_list()[-1] =='aizbraucis'):
        jauns_ieraksts = pd.DataFrame(data={'auto_nr':'-', 'marka_modelis':'-',
                                            'degvielas_veids':'-', 'auto_novietojums':'-','uzlades_stacijas_nr':1234, 'ierasanas_laiks':'-',
                                            'aizbrauksanas_laiks':'-', 'statuss':'tukša vieta', 'timestamp':laiks_tagad()}, index=[0])
        dati = pd.concat([vecie_dati,jauns_ieraksts], ignore_index=True)
    else:
        jauns_ieraksts = vecie_dati.tail(1).copy()  # ja numurs vairs netiek atrasts
        jauns_ieraksts['aizbrauksanas_laiks'] = laiks_tagad()
        jauns_ieraksts['statuss'] = 'aizbraucis'
        jauns_ieraksts['timestamp'] = laiks_tagad()
        dati = pd.concat([vecie_dati,jauns_ieraksts], ignore_index=True)
    
dati.to_csv('datu_tabula.csv', index=False)  # atjauninām datu tabulu
dati
