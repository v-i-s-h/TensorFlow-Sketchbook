# Basic sacred
# Source: https://sacred.readthedocs.io/en/stable/quickstart.html

from sacred import Experiment

ex = Experiment("hello_config")

@ex.config
def my_config():
    recipient = "World"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)