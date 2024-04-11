# ComfyUI wrapper node to test LaVi-Bridge using Diffusers

![image](https://github.com/kijai/ComfyUI-LaVi-Bridge-Wrapper/assets/40791699/8f2cda87-c9e0-4201-94da-42a4d6181871)

# Installing
Either use the Manager and it's install from git -feature, or clone this repo to custom_nodes and run:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Lavi-Bridge-Wrapper\requirements.txt`

The following is autodownloaded:

https://huggingface.co/Kijai/t5-large-encoder-only-bf16/ to `ComfyUI/models/t5_model/``

https://huggingface.co/shihaozhao/LaVi-Bridge/ to `ComfyUI/models/lavibridge`