import os
import time

def remote_merge(path='C:\git_workspace\Tools', name='origin', url='https://github.com/leo914yang/Tools', user='leo914yang', repo='Tools'):
    os.chdir(path)
    time.sleep(3)
    os.system(f'git remote add {name} git@{url}:{user}/{repo}')
    time.sleep(3)
    os.system(f'git fetch {name}')
    time.sleep(3)
    os.system(f'git checkout {name}/master')
    time.sleep(3)
    os.system('git checkout -B master temp')
    time.sleep(3)
    os.system('git branch -d temp')
    time.sleep(3)
    os.system(f'git branch --set-upstream-to={name}/master master')


if __name__=='__main__':
    remote_merge()