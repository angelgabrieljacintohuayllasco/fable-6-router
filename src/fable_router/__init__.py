from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

# Path anclado al repo (no al cwd): Claude Code lanza el MCP server con un cwd
# ajeno al proyecto. encoding="utf-8-sig" traga el BOM que PowerShell 5.1 mete
# con `-Encoding utf8` (con BOM, la primera clave del .env se lee como
# "﻿CLAVE" y nunca matchea).
_env = Path(__file__).resolve().parents[2] / ".env"
if _env.is_file():
    _load_dotenv(_env, encoding="utf-8-sig")
else:  # instalado no-editable: caer a la búsqueda default (cwd hacia arriba)
    _load_dotenv(encoding="utf-8-sig")
