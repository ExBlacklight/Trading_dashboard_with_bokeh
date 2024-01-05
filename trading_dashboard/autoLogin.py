from selenium import webdriver
import time
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import pyotp
from cred import zerodha_id,zerodha_pass,zerodha_key



def Firefox():
    service = webdriver.firefox.service.Service()
    options = FirefoxOptions()
    options.add_argument('-headless')
    options.add_argument('--no-sandbox')
    return webdriver.Firefox(service=service,options=options,service_log_path=os.devnull)


def enterValue(driver,xpath,value=None):
    interactable = driver.find_element("xpath", xpath)
    _ = WebDriverWait(driver,10).until(EC.element_to_be_clickable(interactable))
    interactable.click()
    if value is not None:
        interactable.send_keys(value)


def isVisible(driver,xpath):
    _ = WebDriverWait(driver,45).until(EC.presence_of_element_located((By.XPATH,xpath)))


def waitForURLChange(driver,string):
    _ = WebDriverWait(driver,45).until(EC.url_changes(string))


def formatURL(string):
    words = string.split('?')[1].split('&')
    key = None
    for word in words:
        split_word = word.split('=')
        if split_word[0] == 'status':
            if split_word[1] != 'success':
                raise Exception(f'request token not recieved. please check script')
        if split_word[0] == 'request_token':
            key = split_word[1]
    if key is None:
        raise Exception(f'url succeeded but key not return. check script')
    return key



def get():
    totp = pyotp.TOTP(zerodha_key)
    password = '//*[@id="container"]/div/div/div[2]/form/div[2]/input'
    user = '//*[@id="container"]/div/div/div[2]/form/div[1]/input'
    pass_div = '//*[@id="container"]/div/div/div[2]/form/div[2]'
    url = 'https://kite.trade/connect/login?api_key=9l40nyckacd0aqw4&v=3'
    button = '//*[@id="container"]/div/div/div[2]/form/div[4]/button'
    shield_icon = '//*[contains(concat( " ", @class, " " ), concat( " ", "icon-shield", " " ))]'
    totp_div = '//*[(@id = "userid")]'

    driver = Firefox()
    driver.get(url)
    
    isVisible(driver,user)
    enterValue(driver,user,zerodha_id)
    enterValue(driver,password,zerodha_pass)
    enterValue(driver,button)
    
    isVisible(driver,shield_icon)
    current_url = driver.current_url
    enterValue(driver,totp_div,totp.now())
    
    waitForURLChange(driver,current_url)
    req_url = driver.current_url
    driver.quit()
    print(req_url)
    return formatURL(req_url)

if __name__ == '__main__':
    print(get())