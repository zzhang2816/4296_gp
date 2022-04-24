# 4296 Group Project
This is the artifact of CS4296 Group Project

## prerequisite

```
pip3 install -r requirements.txt
```

## Usage

### Cloud Computing

```
cp -r images/ cloud_computing/client
#on server, in cloud_computing/server
python3 mserver.py -ip 0.0.0.0
#on client, in cloud_computing/client
python3 mclient.py -ip x.x.x.x # server's ip
```

### Edge Computing

```
cp -r images/ edge_computing/client
#on server, in edge_computing/server
python3 mserver.py -ip 0.0.0.0
#on client, in edge_computing/client
python3 mclient.py -ip x.x.x.x # server's ip
```

### Local Distributed Cluster
```
cp -r images/ distributed_computing/server
#on master, in distributed_computing/server
python3 server.py -ip 0.0.0.0
#on slave1, in distributed_computing/client1
python3 client01.py -ip x.x.x.x # master's ip
#on slave2, in distributed_computing/client2
python3 client02.py -ip x.x.x.x # master's ip
```

### DeepThings
Reference to the [DeepThings Repo](https://github.com/SLAM-Lab/DeepThings)