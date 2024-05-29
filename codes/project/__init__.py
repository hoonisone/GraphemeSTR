import platform
os_name = platform.system()

if os_name == "Windows":
    PROJECT_ROOT = "E:/workspace/paddleocr"
elif os_name == "Linux":
    PROJECT_ROOT = "/home"