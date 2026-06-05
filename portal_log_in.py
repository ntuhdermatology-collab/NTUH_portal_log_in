from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from PIL import Image
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoAlertPresentException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import re
import os
import sys
import shutil
import pytesseract


def log_in(person, password, system=0,show=0,headless=0):

    if (system != 0) and (system != 1) and (system != 2) and (system != 3):
        raise ValueError("system can only be int and the value can only be 0,1,2,3")

    if (show != 0) and (show != 1):
        raise ValueError("show can only be int and the value can only be 0,1")

    if (headless != 0) and (headless != 1):
        raise ValueError("headless can only be int and the value can only be 0,1")
    
    
    # 0一般  1門診   2住院
    tesseract_path = shutil.which("tesseract")
    
    if show==1:
        print("Tesseract 路徑:", tesseract_path)
    if tesseract_path is None:
        raise Exception("Tesseract not installed")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    


    # 建立 Chrome Options 物件
    options = Options()
    # 設定 User-Agent
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")
    if headless ==1:
        options.add_argument("--headless")  #無頭模式:不會開啟瀏覽器 debug請去除這行
        options.add_argument("--window-size=1920,1080")  #無頭模式沒有全螢幕  請注意截圖有偏差 為了辨識驗證碼 需更改截圖範圍   

    # 啟動 Edge 並應用自定義的 User-Agent
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.get("https://portal.ntuh.gov.tw/")


    attempt=0
    while (attempt<10):
        element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, "txtUserID")))
        driver.find_element(By.ID, 'txtUserID').clear()
        driver.find_element(By.ID, 'txtUserID').send_keys(person)
        driver.find_element(By.ID, 'txtPass').clear()
        driver.find_element(By.ID, 'txtPass').send_keys(password)


        # 看想要登入什麼系統
        if system ==0:
            1         
        elif system ==1:
            driver.find_element(By.ID, 'rdblQuickMenu_2').click() #門診系統

        elif system ==2:
            driver.find_element(By.ID, 'rdblQuickMenu_3').click() #住院系統
            
        else: #system ==3
            driver.find_element(By.ID, 'rdblQuickMenu_4').click() #急診系統

        # 找到驗證碼圖片的元素
        captcha_element = driver.find_element('id', 'imgVerifyCode')
        captcha_element.screenshot("captcha.png")
        captcha_image = Image.open('captcha.png')
        captcha_image = captcha_image.convert('L')
        captcha_image = captcha_image.point(lambda x: 0 if x < 140 else 255)
        captcha_image = captcha_image.convert('RGB')
        captcha_image.save( "captcha_converted.png" )

        # 使用 Tesseract OCR 識別驗證碼
        captcha_text = pytesseract.image_to_string(captcha_image)
        recognized_captcha = captcha_text.strip()
        modified_recognized_captcha = ''.join(ch for ch in recognized_captcha if ch.isalnum()) # 去除符號
        
        if (len(modified_recognized_captcha)>6):
            modified_recognized_captcha=modified_recognized_captcha[0:6]

        # 輸出 OCR 識別的驗證碼
        if show==1:
            print("識別的驗證碼:", modified_recognized_captcha)

        # 將識別的驗證碼輸入到驗證碼框中
        driver.find_element(By.ID, 'txtVerifyCode').clear()
        driver.find_element(By.ID, 'txtVerifyCode').send_keys(modified_recognized_captcha)
        driver.find_element(By.NAME, 'imgBtnSubmitNew').click()
        try:
        # 看想要登入什麼系統
            if system ==0:
                WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.ID, "btnRefresh_All")))#一般         
            elif system ==1:
                WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.ID, "NTUHWeb1_ShowHideCalender")))#門診系統
            elif system ==2:
                WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.ID, "NTUHWeb1_QueryInPatientPersonAccountControl1_EmpNoCareQueryButton")))#住院系統
            else: #system ==3
                WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.ID, "ctl00_EmerSimplePatientList1_txtChartNo"))) #急診系統
            
            if show==1:
                print ("登入成功")
                os.remove('captcha_converted.png')
            os.remove('captcha.png')
            break
        except:
            if show==1:
                print("辨認錯誤")
            if attempt<10:
                attempt=attempt+1
                continue
            else:
                raise RuntimeError ("repeat attempt > 9, fail logged in")

    number_of_tabs = len(driver.window_handles)
    window_handles = driver.window_handles
    if (number_of_tabs > 1):
           for j in range(1, len(window_handles)):
                  driver.switch_to.window(window_handles[j])
                  driver.close()
    driver.switch_to.window(window_handles[0])
    return driver

