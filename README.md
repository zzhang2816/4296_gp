# 4296_gp

### prerequisite

```
pip install -r requirements.txt
```

### Usage

Cloud computing

```
mv src/ cloud_computing/server
mv image/ cloud_computing/client
python3 mserver.py -ip xx
python3 mclient.py -ip xx
```

Edge computing

```
mv src/ edge_computing/server
mv image/ edge_computing/client
python3 mserver.py -ip 0.0.0.0
python3 mclient.py -ip xx
```

