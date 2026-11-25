Deliverable 3: Final Report - LaTeX source
==========================================

Contents
  Final_Report.tex   LaTeX source of the final report
  Final_Report.pdf   the compiled report
  media/             all images used by the report (IBA logo, EDA figures, OCTree diagram,
                     result figures and the three application screenshots)

How to compile
  Overleaf: upload this folder (or the zip), open Menu -> Compiler and choose XeLaTeX, then
            click Recompile.
  Locally:  xelatex Final_Report.tex   (run it 2-3 times so the contents, table and figure
            numbers are up to date)

Changing the GitHub link
  The link on the title page is set near the top of Final_Report.tex:
      \newcommand{\githublink}{\url{...}}
  Replace the address between the inner braces and recompile.
