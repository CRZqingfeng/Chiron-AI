@echo off
REM Stop Chiron-AI: backend/frontend are in the two cmd windows (Ctrl+C there).
REM This stops the database containers.
cd /d %~dp0
docker-compose stop
echo Database containers stopped. Close the backend/frontend windows manually if still open.
pause
