# Templates

Plantillas reutilizables para los informes en LaTeX del curso.

## Deliverables

| File | Description |
| :--- | :---------- |
| `Informe.tex` | Plantilla de informe (`article`, 11 pt, en español con `babel`) con `amsmath`, `booktabs`, `listings` para código Python, `pgfplots` y bibliografía con `biblatex`. |
| `referencias.bib` | Archivo de referencias que `Informe.tex` carga con `\addbibresource`. |

## Execution

Para usarla, copiar ambos archivos a la carpeta del informe y compilar con `biber` (el preámbulo usa `backend=biber`):

```bash
pdflatex Informe.tex
biber Informe
pdflatex Informe.tex
pdflatex Informe.tex
```
