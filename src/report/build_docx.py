"""Gera relatorio.docx a partir de relatorio.tex (o .tex é a fonte; não edite o .docx).

Uso, na raiz do repositório:
    uv run --with pypandoc_binary python src/report/build_docx.py
"""

import os
import re
import tempfile
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent


def main():
    tex = (HERE / 'relatorio.tex').read_text(encoding='utf-8')
    # O Word não lê PDF: usa as versões PNG das figuras.
    tex = re.sub(r'\\includegraphics(\[[^\]]*\])?\{(figures/[^}.]+)\}', r'\\includegraphics\1{\2.png}', tex)
    # Tabelas embutidas e limpas do que o pandoc não interpreta (\cmidrule, @{} no \multicolumn).
    def inline_table(match):
        table = (HERE / f'{match.group(1)}.tex').read_text(encoding='utf-8')
        table = re.sub(r'\\cmidrule(\([a-z]*\))?\{[^}]*\}', '', table)
        return table.replace('{@{}l}', '{l}')
    tex = re.sub(r'\\input\{(tables/[^}]+)\}', inline_table, tex)
    # \paragraph vira título numerado no Word; usamos negrito no início do parágrafo.
    tex = re.sub(r'\\paragraph\{([^}]*)\}', r'\\textbf{\1}', tex)
    tex = tex.replace(r'\setlength\itemsep{0pt}', '')

    os.chdir(HERE)  # \input{tables/...} e figures/ são relativos a esta pasta
    with tempfile.NamedTemporaryFile('w', suffix='.tex', dir=HERE, delete=False, encoding='utf-8') as tmp:
        tmp.write(tex)
    try:
        pypandoc.convert_file(
            tmp.name, 'docx', format='latex', outputfile=str(HERE / 'relatorio.docx'),
            extra_args=['--citeproc', '--bibliography=refs.bib', '--metadata=lang:pt-BR',
                        '--resource-path=.', '--number-sections'],
        )
    finally:
        os.unlink(tmp.name)
    print('relatorio.docx gerado')


if __name__ == '__main__':
    main()
