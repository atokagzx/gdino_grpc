import os, sys  
_pb2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pb2_include')
sys.path.append(_pb2_path)
print(f"appending {_pb2_path} to sys.path")

from __init__ import *
sys.path.remove(_pb2_path)
print(f"removing {_pb2_path} from sys.path")