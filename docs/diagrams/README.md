# Diagrams

| File | What it shows |
| --- | --- |
| [architecture.drawio](architecture.drawio) / [architecture.svg](architecture.svg) | Modules around the query loop |
| [query_loop.drawio](query_loop.drawio) / [query_loop.svg](query_loop.svg) | Turn workflow |
| [tool_pipeline.drawio](tool_pipeline.drawio) / [tool_pipeline.svg](tool_pipeline.svg) | Validate → hook → permit → execute |
| [bootstrap.drawio](bootstrap.drawio) / [bootstrap.svg](bootstrap.svg) | Process init |

Open a `.drawio` file at [https://app.diagrams.net/](https://app.diagrams.net/) (File → Open from device). After edits: File → Export as → SVG into this folder, same basename.

If [draw.io desktop](https://github.com/jgraph/drawio-desktop/releases) is installed:

```powershell
& "C:\Program Files\draw.io\draw.io.exe" -x -f svg -o architecture.svg architecture.drawio
```
