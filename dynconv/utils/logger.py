""" Logger for printing """

# Settings
interval = 50


# Logger
loggers = {}
step = 0

def add(name, val):
    global loggers, step
    loggers[name] = val

def tick(file):
    global loggers, step
        # print(f'\n $$$ Step {step}')
    print('', file=file)
    for name in sorted(loggers):
        val = loggers[name]
        if val is not None:
            print('$ ',str(name).ljust(15), ':', val, file=file)

def reset():
    global loggers, step
    loggers = {}

import torch
