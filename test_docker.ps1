# Script PowerShell pour tester le container Docker ForestGaps
# Usage: .\test_docker.ps1

Write-Host "=== Test Container ForestGaps ===" -ForegroundColor Cyan

# 1. Vérifier que le container tourne
Write-Host "`n1. Vérification du container..." -ForegroundColor Yellow
docker ps --filter "name=forestgaps-main" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# 2. Tester l'import forestgaps
Write-Host "`n2. Test import forestgaps..." -ForegroundColor Yellow
docker exec forestgaps-main python -c "import forestgaps; print('✓ ForestGaps OK')"

# 3. Vérifier la structure /app
Write-Host "`n3. Structure /app..." -ForegroundColor Yellow
docker exec forestgaps-main ls -la /app/ | Select-String "^d"

# 4. Vérifier si scripts est monté
Write-Host "`n4. Vérification scripts..." -ForegroundColor Yellow
docker exec forestgaps-main test -d /app/scripts && Write-Host "✓ /app/scripts existe" -ForegroundColor Green || Write-Host "✗ /app/scripts manquant" -ForegroundColor Red

# 5. Lister les scripts si le dossier existe
docker exec forestgaps-main sh -c "if [ -d /app/scripts ]; then ls -la /app/scripts/*.py 2>/dev/null || echo 'Pas de fichiers .py'; fi"

# 6. Vérifier forestgaps.environment
Write-Host "`n5. Test forestgaps.environment..." -ForegroundColor Yellow
docker exec forestgaps-main python -c "from forestgaps.environment import setup_environment; env = setup_environment(); print('✓ Environment OK:', type(env).__name__)"

Write-Host "`n=== Tests terminés ===" -ForegroundColor Cyan
