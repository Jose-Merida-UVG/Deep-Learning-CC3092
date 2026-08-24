# Neovim Cheat Sheet

Leader = `<space>`

## General

| Key | Action |
|---|---|
| `<leader>w` | Save (silent) |
| `<leader>q` | Quit window |
| `<leader>Q` | Quit everything (discard changes) |
| `<C-h/j/k/l>` | Move between splits (works from terminal-mode too) |
| `[d` / `]d` | Prev/next diagnostic |
| `<leader>d` | Line diagnostics |
| `<leader>ca` | Code action |
| `<leader>rn` | Rename symbol |
| `gd` / `gr` / `K` | Go to definition / references / hover |

## Buffers & files

| Key | Action |
|---|---|
| `<S-l>` / `<S-h>` | Next/prev buffer |
| `<leader>bd` | Close buffer |
| `<leader>e` | File explorer (`i` to search-filter the list) |
| `<leader>ff` | Find file |
| `<leader>fg` | Grep |
| `<leader>fh` | Recent files |
| `<leader>fb` | Find buffer |

## Claude Code

| Key | Action |
|---|---|
| `<leader>ac` | Toggle panel |
| `<leader>af` | Focus panel |
| `<leader>as` (visual) | Send selection |
| `<leader>aa` / `<leader>ad` | Accept / reject diff |

## Ollama ghost-text completion

| Key | Action |
|---|---|
| `<A-A>` | Accept full suggestion |
| `<A-a>` | Accept one line |
| `<A-z>` | Accept N lines |
| `<A-[>` / `<A-]>` | Cycle prev/next suggestion |
| `<A-e>` | Dismiss |
| `<leader>am` | Toggle auto-trigger on/off |

## Jupyter (molten.nvim + notebook-navigator.nvim)

| Key | Action |
|---|---|
| `<leader>ji` | Start kernel for this buffer (`:MoltenInit`) |
| `]h` / `[h` | Next/prev cell |
| `<leader>jc` | Run current cell |
| `<leader>jn` | Run cell, then move to next |
| `<leader>jr` (visual) | Run selection |
| `<leader>ja` / `<leader>jA` | New cell below / above |
| `<leader>js` | Split cell at cursor |
| `<leader>jo` / `<leader>jh` | Enter / hide output window |
| `<leader>jd` | Delete cell output |

### Running against a project `.venv`

`:MoltenInit` picks from registered Jupyter kernels, not venvs directly.
Register a venv as a kernel once per project:

```bash
cd ~/your-project
source .venv/bin/activate
pip install ipykernel   # if not already present
python -m ipykernel install --user --name my-project --display-name "My Project"
```

Then in nvim: `:MoltenInit` (picker) or `:MoltenInit my-project` (direct).

Check what's registered: `jupyter kernelspec list`

Give each project's venv a **distinct** `--name` -- reusing a name silently
overwrites the previous kernelspec.
