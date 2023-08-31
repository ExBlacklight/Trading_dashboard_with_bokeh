from selenium import webdriver
import time
from cred import zerodha_id,zerodha_pass,zerodha_pin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def Chrome():
    return webdriver.Chrome()

def Firefox():
    return webdriver.Firefox()

def get():
    driver = Firefox()
    password = '//*[@id="container"]/div/div/div[2]/form/div[2]/input'
    user = '//*[@id="container"]/div/div/div[2]/form/div[1]/input'
    pass_div = '//*[@id="container"]/div/div/div[2]/form/div[2]'
    url = 'https://kite.trade/connect/login?api_key=9l40nyckacd0aqw4&v=3'
    driver.get(url)
    while True:
        try:
            user_elem = driver.find_element_by_xpath(user)
            user_elem.send_keys(zerodha_id)
            break
        except:
            pass

    current_url = driver.current_url
    while True:
        try:
            pass_div_elem = driver.find_element_by_xpath(pass_div)
            pass_elem = driver.find_element_by_xpath(password)
            pass_div_elem.click()
            pass_elem.send_keys(zerodha_pass)
            but = '//*[@id="container"]/div/div/div[2]/form/div[4]/button'
            submit = driver.find_element_by_xpath(but)
            submit.click()
            break
        except:
            pass
    
    while current_url == driver.current_url:
        time.sleep(0.5)
    got_it = False
    while got_it == False:
        try:
            q = driver.current_url
            w = q.split('?')[1].split('&')
            for i in w:
                o = i.split('=')
                if o[0] == 'status':
                    if o[1] != 'success':
                        print('request token not recieved. please check script')
                    else:
                        got_it = True
            
        except:
            pass

    q = driver.current_url
    print(q)
    w = q.split('?')[1].split('&')
    for i in w:
        o = i.split('=')
        if o[0] == 'request_token':
            result = o[1]

    driver.quit()
    return result

if __name__ == '__main__':
    get()