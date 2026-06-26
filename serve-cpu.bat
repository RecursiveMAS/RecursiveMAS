@echo off
setlocal

REM Load variables from .env (skips comment lines starting with #)
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

echo.
echo  RecursiveMAS ^| Web UI ^| CPU mode
echo  ----------------------------------------
echo  Open http://localhost:7860 in your browser
echo  Press Ctrl+C to stop.
echo.

docker run --rm -p 7860:7860 ^
  -e HF_TOKEN=%HF_TOKEN% ^
  -e TAVILY_API_KEY=%TAVILY_API_KEY% ^
  -v recursivemas_hf_cache:/hf_cache ^
  --entrypoint python ^
  recursivemas-serve ^
  serve.py --host 0.0.0.0 --port 7860

endlocal
