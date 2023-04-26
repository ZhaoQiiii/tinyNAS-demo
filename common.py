import json
from tqdm  import tqdm

import time
import logging
import requests
import os 
import threading
from requests.adapters import HTTPAdapter
from pynas import NasClient

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def daytime():
    return 60

def thread_process(path):
    while True:
        
        files = os.listdir(path)
        cur_time = time.time() 
        for file in files:
            filename = os.path.join(path, file)
            c_time = os.path.getctime(filename) 
            if (cur_time - c_time) > daytime() :
                try:
                    logger.info(f'going to delete f{filename}')
                    os.remove(filename)
                except:
                    pass
        time.sleep(daytime())
    
DEFAULT_TIME_OUT = 60*60*24

class Client():
    def __init__(self, logger):
        self.sess = requests.Session()
        self.sess.mount('http://', HTTPAdapter(max_retries=3))
        self.sess.mount('https://', HTTPAdapter(max_retries=3))
        self.logger = logger
        self.client = NasClient(self.logger)
        self.subpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_models')
        logger.info(f'temp save path = {self.subpath}')
        os.makedirs(self.subpath, exist_ok=True)
        self.thread = threading.Thread(target = thread_process, args=(self.subpath,))
        self.thread.setDaemon(True)
        self.thread.start()

    def __del__(self,):
        self.thread.join()

    def commit(self, data, timeout = DEFAULT_TIME_OUT):
        try:
            self.logger.info('commit request {}'.format(json.dumps(data)))
            request_id = self.client.put(data)
            start_time = time.time()
            while True:
                payload = self.client.try_get(request_id)
                if payload :
                    return  payload['url'].replace('http://', 'https://')

                cost_time = int(time.time() - start_time)
                if cost_time >= timeout:
                    self.logger.warning(f'cost_time= {cost_time}s has exceeded max timeout =  {timeout}s, and you can set bigger timeout value for large task')
                    return None
                time.sleep(10)
        except:
            self.logger.info('commit error')
            return None

    def download(self, url, stream=True):
        filename = url.split('?')[0].split('%')[-1].split('%')[-1][2:]
        download_targz = os.path.join(self.subpath, filename)
        
        try:
            resp = self.sess.get(url, stream=stream)
            file_size = int(resp.headers['content-length'])
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, ascii=True, desc=download_targz) as bar:
                with open(download_targz, 'wb') as fp:
                    for chunk in resp.iter_content(chunk_size=512):
                        if chunk:
                            fp.write(chunk)
                            bar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            self.logger.error(e)
            return None

        return download_targz
        
def adjust_data_range(budget_model_size, budget_flops):
    
    budget_model_size  =int(budget_model_size  * 1e6)
    budget_flops       =int(budget_flops       * 1e6)
    
    return budget_model_size, budget_flops

client = Client(logger)

title =  "TinyNAS"
