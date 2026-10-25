"""Read-only compatibility probe for an existing environment."""
import importlib
import json

result={}
for name in ('torch','diffusers','transformers','accelerate','huggingface_hub'):
    module=importlib.import_module(name);result[name]=module.__version__
try:
    from diffusers import QwenImage21Pipeline
    import inspect
    result['pipeline']=str(inspect.signature(QwenImage21Pipeline.__call__))
except (ImportError,AttributeError,RuntimeError) as error:
    result['pipeline_error']=repr(error)
print(json.dumps(result,indent=2),flush=True)
