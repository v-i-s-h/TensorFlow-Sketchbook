This folder contains tools and code snippets for managing projects.

## Tools

### [Zookeeper](https://github.com/plumerai/zookeeper)
```
```

### [Sacred](https://github.com/IDSIA/sacred)
  - [Docs](https://sacred.readthedocs.io/en/stable/)
  - Use [Omniboard](https://github.com/vivekratnavel/omniboard) for visualization 

#### Usage
- Install scared, pymongo, nodejs, omniboard inside your environment. 
```
    pip install sacred
    pip install pymongo
    conda install -c conda-forge nodejs
    npm install -g omniboard
```

- Create a configuration file ([Sample](./sacred/zoo/MongoDB/mongod.conf))

- Start MongoDB as `mongod --config <config file>`
    - Example: From `./sacred/zoo/MongoDB`, run `mongod --config ./mongod.conf`

- Start Omniboard `omniboard -m hostip:port:data`
    - Example:
        `omniboard -m 127.0.0.1:27017:test_data`

Run experiment from `./sacred/` as 
```
    python 01_keras.py with activation='sigmoid'
    python 01_keras.py with activation='tanh'
    python 01_keras.py with dropout=0.10
```