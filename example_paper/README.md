# CVPR 2026 Paper Template

## Structure

This paper template follows a modular structure to facilitate collaboration and organization:

```
paper/
├── paper.tex              # Main file (compile this)
├── sections/              # Modular sections
│   ├── packages.tex       # Package imports
│   ├── commands.tex       # Custom commands and macros
│   ├── title.tex          # Title and authors
│   ├── abstract.tex       # Abstract
│   ├── introduction.tex   # Introduction section
│   ├── related_work.tex   # Related Work section
│   ├── method.tex         # Method section
│   ├── experiments.tex    # Experiments section
│   ├── conclusion.tex     # Conclusion section
│   ├── acknowledgments.tex # Acknowledgments
│   └── references.tex     # References/Bibliography
└── figures/               # Store figures here (create when needed)
```

## How to Use

1. **Compile**: Run LaTeX on `paper.tex` (the main file)
2. **Edit sections**: Modify individual files in `sections/` directory
3. **Add packages**: Edit `sections/packages.tex`
4. **Add custom commands**: Edit `sections/commands.tex`
5. **Add figures**: Place images in `figures/` directory and reference them

## Benefits of Modular Structure

- **Easier collaboration**: Multiple authors can work on different sections simultaneously
- **Better organization**: Each section has its own file
- **Faster compilation**: Can comment out sections during development
- **Version control friendly**: Git diffs are cleaner when changes are in separate files
- **Reusability**: Easy to copy sections to other papers

## Compilation

To compile the paper:
```bash
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

Or use your LaTeX editor's build command.
