#!/usr/bin/env python3
"""
Generates paper/main.tex from docs/manuscript_draft_tifs.md
Matches the exact style, preamble, and elegance of prism/paper/main.tex.
"""

import os
import re

HEADER = r"""\documentclass[10pt,twocolumn,letterpaper]{article}

\usepackage{iftex}
\ifPDFTeX
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
\else
  \usepackage{fontspec}
\fi
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{microtype}
\usepackage{cite}
\usepackage{subcaption}
\usepackage{tabularx}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=teal,
    urlcolor=magenta
}

% Float placement parameters
\renewcommand{\topfraction}{0.95}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{dbltopnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{6}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}

\title{\textbf{Continuous-Time Quantum Walk Graph Neural Networks for Evading Adversarial Topological Camouflage in Network Intrusion Detection}}

\author{
    \textbf{Nguyen Trong Luu}, \textbf{Hieu Minh Pham}, \textbf{Anh Quoc Phan}, \textbf{Cho Xuan Do}\thanks{Corresponding author: \texttt{chodx@ptit.edu.vn}} \\
    \textit{Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam}
}
\date{}

\begin{document}

\maketitle
"""

print("[*] Generating publication-grade LaTeX manuscript...")
