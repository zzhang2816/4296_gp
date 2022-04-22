# 4296_gp

### prerequisite

```
pip install -r requirements.txt
```

### Usage

Cloud computing

```
mv image/ cloud_computing/client
python3 mserver.py -ip xx
python3 mclient.py -ip xx
```

Edge computing

```
mv image/ edge_computing/client
python3 mserver.py -ip 0.0.0.0
python3 mclient.py -ip xx
```

Distributed Edge computing

```
mv image/ distributed_computing/client
python3 server/mserver.py -ip 0.0.0.0
python3 client1/client01.py -ip xx
python3 client2/client02.py -ip xx
```
