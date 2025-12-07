# REST-MEX 2025 – CIMAT Team

Este repositorio contiene el código fuente de nuestro paper enviado a IberLEF 2025.  
Utiliza la plantilla oficial de CEUR-WS.

## Estructura de carpetas

- `main.tex`       – Archivo principal de LaTeX  
- `refs.bib`       – Archivo BibTeX con las referencias  
- `ceurart.cls`    – Clase oficial de CEUR-WS  
- `ceur-ws-logo.png` – Logotipo para el pie “CEUR Workshop Proceedings”  
- `figures/`       – Imágenes utilizadas en el documento  
- `build/`         – Carpeta para el PDF compilado  
- `LICENSE.txt`    – Licencia (CC BY 4.0)  

## Cómo compilar

1. Verifica que `ceurart.cls` y `ceur-ws-logo.png` estén en la raíz.  
2. En Overleaf: selecciona `main.tex` como archivo principal y `Recompile`.  
3. En local (Linux/macOS/Windows):
   ```bash
   pdflatex main.tex
   bibtex main      # o biber main, según tu flujo de bibliografía
   pdflatex main.tex
   pdflatex main.tex
