@echo off
REM Chiron-AI daily startup script
REM 1) start database containers  2) backend  3) wait until backend is ready  4) frontend
cd /d %~dp0

REM ★ 后端端口从 .env 的 UVICORN_PORT 读取，避免与 run.py / vite proxy 不一致。
REM   曾因三处各写各的：run.py 用 19999、start.bat 死等 9999，结果卡在 [3/4] 永远等不到。
set "BACKEND_PORT=19999"
for /f "usebackq tokens=2 delims==" %%b in (`findstr /b /c:"UVICORN_PORT" .env 2^>nul`) do set "BACKEND_PORT=%%b"
set "BACKEND_PORT=%BACKEND_PORT: =%"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=19999"

echo [1/4] Starting database containers (postgres/redis/etcd/minio/milvus)...
docker-compose up -d postgres redis etcd minio minio-app standalone

echo [2/4] Starting backend in a new window...
start "Chiron-AI Backend" cmd /k "venv\Scripts\activate && python run.py"

echo [3/4] Waiting for backend on 127.0.0.1:%BACKEND_PORT% to accept requests...
REM Backend loads LangChain/LangGraph and connects to Postgres/Milvus: cold start takes 20-60s.
REM Starting the frontend too early makes Vite proxy /api/v1/base/refresh fail with ECONNREFUSED.
set /a _try=0
:wait_backend
curl -s -o nul -m 2 http://127.0.0.1:%BACKEND_PORT%/api/v1/base/registration_enabled
if not errorlevel 1 goto backend_ready
set /a _try+=1
if %_try% geq 90 goto backend_timeout
timeout /t 2 /nobreak >nul
goto wait_backend

:backend_ready
echo Backend is up (waited about %_try% x 2s).
echo [4/4] Starting frontend in a new window...
start "Chiron-AI Frontend" cmd /k "cd web && pnpm dev"
goto done

:backend_timeout
echo.
echo WARNING: backend did not respond within about 3 minutes. Starting frontend anyway.
echo          Check the "Chiron-AI Backend" window for errors, then refresh the page.

:done
echo.
echo Done!
echo   Frontend : http://localhost:3200
echo   API docs : http://localhost:%BACKEND_PORT%/docs
echo   First login: admin / see .env INITIAL_ADMIN_PASSWORD
echo.
echo To stop everything: run stop.bat (or close the two windows)
pause
