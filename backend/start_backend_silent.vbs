' start_backend_silent.vbs
' Launches DeepGuard AI FastAPI backend silently in the background (no visible command window)
Set WshShell = CreateObject("WScript.Shell")
strPath = WshShell.CurrentDirectory

' Run uvicorn on 0.0.0.0:8000 with window hidden (0)
WshShell.Run "cmd /c python -m uvicorn main:app --host 0.0.0.0 --port 8000", 0, False
Set WshShell = Nothing
