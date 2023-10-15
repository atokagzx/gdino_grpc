## This directory contains the Protocol Buffers include files used by the gRPC.
***.py** files are pregenerated from the ***.proto** files in the parent directory by default.  
If you want to regenerate them, run the following command in the parent directory:
```bash
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. *.proto
```
Make sure you have installed the `grpcio-tools` package.