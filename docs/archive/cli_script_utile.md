## Compacter le disque Docker 
# 1. Arrêter Docker Desktop complètement
# (Clic droit sur l'icône → Quit)

# 2. Compacter le disque virtuel
# Ouvrir PowerShell en ADMIN
Optimize-VHD -Path "C:\Users\<TON_USER>\AppData\Local\Docker\wsl\disk\docker_data.vhdx" -Mode Full

# OU si Hyper-V n'est pas dispo :
wsl --shutdown
diskpart
# Dans diskpart :
select vdisk file="C:\Users\<TON_USER>\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
compact vdisk
exit

# 3. Nettoyer les images/volumes inutiles AVANT de compacter
docker system prune -a --volumes
##