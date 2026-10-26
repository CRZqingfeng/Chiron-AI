@echo off
REM Chiron-AI daily startup script
REM 1) start database containers  2) backend  3) wait until backend is ready  4) frontend
cd /d %~dp0

echo [1/4] Starting database containers (postgres/redis/etcd/minio/milvus)...
docker-compose up -d postgres redis etcd minio minio-app standalone

echo [2/4] Starting backend in a new window...
start "Chiron-AI Backend" cmd /k "venv\Scripts\activate && python run.py"

echo [3/4] Waiting for backend on 127.0.0.1:9999 to accept requests...
REM Backend loads LangChain/LangGraph and connects to Postgres/Milvus: cold start takes 20-60s.
REM Starting the frontend too early makes Vite proxy /api/v1/base/refresh fail with ECONNREFUSED.
set /a _try=0
:wait_backend
curl -s -o nul -m 2 http://127.0.0.1:9999/api/v1/base/registration_enabled
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
echo   API docs : http://localhost:9999/docs
echo   First login: admin / see .env INITIAL_ADMIN_PASSWORD
echo.
echo To stop everything: run stop.bat (or close the two windows)
pause
