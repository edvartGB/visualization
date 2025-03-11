%% Identification part:

\NeedsTeXFormat{LaTeX2e}[1994/12/01]

\ProvidesClass{uiobrev}[2023/09/12 v1.05 UiO document class]

%% Initial code part:

\RequirePackage{etoolbox, xifthen}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{tabularx}

\newtoggle{uiobrev@marks}

%% Options declaration part:

\DeclareOption{marks}{\toggletrue{uiobrev@marks}}
\DeclareOption{nomarks}{\togglefalse{uiobrev@marks}}

\DeclareOption{10pt}{\def \uiobrev@size {10pt}}
\DeclareOption{11pt}{\def \uiobrev@size {11pt}}
\DeclareOption{12pt}{\def \uiobrev@size {12pt}}

\DeclareOption{norsk}{\def \uiobrev@lang {bm}}
\DeclareOption{nynorsk}{\def \uiobrev@lang {nn}}
\DeclareOption{UKenglish}{\def \uiobrev@lang {eng}}
\DeclareOption{USenglish}{\def \uiobrev@lang {us}}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{letter}}

%% Options execution part:

\ExecuteOptions{marks, norsk, 11pt}
\ProcessOptions \relax

%% Package loading part:

\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}

\LoadClass[\uiobrev@size]{letter}

%% Language adaption:

\newcommand{\uiobrev@matnat@bm}{Det matematisk-natur\-viten\-skapelige fakultet}
\newcommand{\uiobrev@matnat@nn}{Det matematisk-natur\-viten\-skaplege fakultet}
\newcommand{\uiobrev@matnat@eng}{Faculty of Mathematics and Natural Sciences}
\newcommand{\uiobrev@matnat@us}{Faculty of Mathematics and Natural
  Sciences}

\newcommand{\uiobrev@ifi@bm}{Institutt for informatikk}
\newcommand{\uiobrev@ifi@nn}{Institutt for informatikk}
\newcommand{\uiobrev@ifi@eng}{Department of Informatics}
\newcommand{\uiobrev@ifi@us}{Department of Informatics}

\newcommand{\uiobrev@phone@bm}{Tlf}
\newcommand{\uiobrev@phone@nn}{Tlf}
\newcommand{\uiobrev@phone@eng}{Phone}
\newcommand{\uiobrev@phone@us}{Phone}

\newcommand{\uiobrev@email@bm}{E-post}
\newcommand{\uiobrev@email@nn}{E-post}
\newcommand{\uiobrev@email@eng}{E-mail}
\newcommand{\uiobrev@email@us}{E-mail}

\newcommand{\uiobrev@org@bm}{Org.nr}
\newcommand{\uiobrev@org@nn}{Org.nr}
\newcommand{\uiobrev@org@eng}{Org no}
\newcommand{\uiobrev@org@us}{Org no}

\newcommand{\uiobrev@mailaddr@bm}{Postadresse}
\newcommand{\uiobrev@mailaddr@nn}{Postadresse}
\newcommand{\uiobrev@mailaddr@eng}{Postal address}
\newcommand{\uiobrev@mailaddr@us}{Postal address}

\newcommand{\uiobrev@visitaddr@bm}{Bes\o ksadresse}
\newcommand{\uiobrev@visitaddr@nn}{Vitjingsadresse}
\newcommand{\uiobrev@visitaddr@eng}{Visiting address}
\newcommand{\uiobrev@visitaddr@us}{Visiting address}

\newcommand{\uiobrev@dato@bm}{Dato}
\newcommand{\uiobrev@dato@nn}{Dato}
\newcommand{\uiobrev@dato@eng}{Date}
\newcommand{\uiobrev@dato@us}{Date}

\newcommand{\uiobrev@ourref@bm}{V\aa r ref}
\newcommand{\uiobrev@ourref@nn}{V\aa r ref}
\newcommand{\uiobrev@ourref@eng}{Our ref}
\newcommand{\uiobrev@ourref@us}{Our ref}

\newcommand{\uiobrev@yourref@bm}{Deres ref}
\newcommand{\uiobrev@yourref@nn}{Deres ref}
\newcommand{\uiobrev@yourref@eng}{Your ref}
\newcommand{\uiobrev@yourref@us}{Your ref}

%% Letter information options:

\newcommand{\countersign}[1]{\def \uiobrev@countersign {#1}}
\renewcommand{\date}[1]{\def \uiobrev@today {#1}}
\newcommand{\mycellphone}[1]{\telephone{#1}} % For backwards compatibility
\newcommand{\mymobilephone}[1]{\telephone{#1}} % For backwards compatibility
\newcommand{\myemail}[1]{\def \uiobrev@email@info {#1}}
\newcommand{\ourdept}[1]{\def \uiobrev@dept@info {#1}}
\newcommand{\ourfac}[1]{\def \uiobrev@fac@info {#1}}
\newcommand{\ourfax}[1]{}  % For backwards compatibility
\newcommand{\ourgroup}[1]{\def \uiobrev@group@info {#1}}
\newcommand{\ourorgno}[1]{\def \uiobrev@org@info {#1}}
\newcommand{\ourref}[1]{\def \uiobrev@ourref {#1}}
\newcommand{\ourvisitaddress}[1]{\def \uiobrev@visit@info {#1}}
\newcommand{\ourwww}[1]{\def \uiobrev@www@info {#1}}
\newcommand{\yourref}[1]{\def \uiobrev@yourref {#1}}

% Default values:

\countersign{}
\date{\today}
\myemail{}
\ourdept{\csname uiobrev@ifi@\uiobrev@lang\endcsname}
\ourfac{\csname uiobrev@matnat@\uiobrev@lang\endcsname}
\ourgroup{}
\ourorgno{}
\ourref{}
\ourvisitaddress{}
\ourwww{}
\telephone{}
\yourref{}

%% Redefine the 'letter' environment, \operning and \closing:

\renewenvironment{letter}[1]{\newpage
   \interlinepenalty=200  % from the original 'letter' class
   \@processto{\leavevmode\ignorespaces #1}
   \uiobrev@letterhead \uiobrev@recipient
   \iftoggle{uiobrev@marks}{\uiobrev@fold@marks}{}
   \vspace*{4.5cm}}
  {\stopletter\@@par\pagebreak\@@par}

\newcommand{\uiobrev@letterhead}{%
  \begin{textblock*}{2cm}(1cm,1cm)
    \includegraphics[width=2cm]{uiobrev-segl}
  \end{textblock*}%
  \begin{textblock*}{15cm}(3.4cm,1.48cm)
    \includegraphics[height=1.04cm]{uiobrev-navn-\uiobrev@lang}
  \end{textblock*}}

\newcommand{\uiobrev@recipient}{%
  \begin{textblock*}{10.5cm}(2cm,4.6cm)
    \textbf{\toname}
    \ifcsempty{toaddress}{}{\\ \toaddress}
  \end{textblock*}}

\newcommand{\uiobrev@fold@marks}{%
  \begin{textblock*}{1cm}(0cm,9.9cm)
    \rule{1cm}{0.5pt}
  \end{textblock*}
  \begin{textblock*}{1cm}(0cm,19.8cm)
    \rule{1cm}{0.5pt}
  \end{textblock*}}

\newcommand{\uiobrev@info@setup}{\setlength{\parskip}{3pt}%
  \fontsize{9}{11pt}\selectfont \raggedright}
 
\newcommand{\uiobrev@info}{%
  \begin{textblock*}{17cm}(2cm,26.7cm)
    \rule{17cm}{0.5pt}
  \end{textblock*}
  \begin{textblock*}{4.1cm}(2cm,27cm)
    \uiobrev@info@setup
    \textbf{\uiobrev@fac@info}\par
    \uiobrev@dept@info\par
    \uiobrev@group@info\par
  \end{textblock*}
  \begin{textblock*}{4.1cm}(6.3cm,27cm)
    \ifcsempty{fromaddress}{}{%
      \uiobrev@info@setup
      \csname uiobrev@mailaddr@\uiobrev@lang\endcsname:\par
      \fromaddress}
  \end{textblock*}
  \begin{textblock*}{4.1cm}(10.6cm,27cm)
    \ifcsempty{uiobrev@visit@info}{}{%
      \uiobrev@info@setup
      \csname uiobrev@visitaddr@\uiobrev@lang\endcsname:\par
      \uiobrev@visit@info}
  \end{textblock*}
  \begin{textblock*}{4.1cm}(14.9cm,27cm)
    \uiobrev@info@setup \setlength{\parskip}{0pt}
    \ifcsempty{telephonenum}{}{%
      \csname uiobrev@phone@\uiobrev@lang\endcsname:
      \telephonenum\par}
    \ifcsempty{uiobrev@email@info}{}{%
      \csname uiobrev@email@\uiobrev@lang\endcsname:
      \uiobrev@email@info\par}
    \ifcsempty{uiobrev@www@info}{}{%
      \uiobrev@www@info\par}
    \ifcsempty{uiobrev@org@info}{}{%
      \csname uiobrev@org@\uiobrev@lang\endcsname:
      \uiobrev@org@info\par}
  \end{textblock*}}

\renewcommand{\opening}[1]{\par
  \begin{minipage}{0.8\textwidth}
    {\fontsize{9}{11pt}\selectfont
     \noindent\begin{tabular}{@{}l@{\hspace{0.3em}}l@{}}
        \csname uiobrev@dato@\uiobrev@lang\endcsname:& \uiobrev@today\\
      \ifcsempty{uiobrev@yourref}{}
        {\csname uiobrev@yourref@\uiobrev@lang\endcsname:& \uiobrev@yourref\\}
      \ifcsempty{uiobrev@ourref}{}
        {\csname uiobrev@ourref@\uiobrev@lang\endcsname:& \uiobrev@ourref\\}
     \end{tabular}}\par
    \vspace{3ex}
    \textbf{#1}
  \end{minipage}\par
  \nobreak}

% \setlength{\longindentation}{0cm}

\renewcommand{\closing}[1]{\par
  \nobreak\vspace{\parskip}%
  \noindent
  \ifcsempty{fromsig}
    {\ignorespaces #1\\ \fromname}
    {\ignorespaces #1\\[6ex] \fromsig}\par
  \ifcsempty{uiobrev@countersign}{}
    {\nobreak\vspace{3ex}%
      \noindent\hspace*{0.5\textwidth}\begin{tabular}{@{}c@{}}
        \uiobrev@countersign
      \end{tabular}\par}
}

%% Page layout

\RequirePackage[a4paper, vmargin=3.5cm, hmargin=2cm]{geometry}
  %% Must be used after 'letter' class has been loaded.

\raggedright

%% Info at the bottom is made with a special page style:
\newcommand{\ps@uiobrev}{%
  \def \@evenhead {}%
  \def \@oddhead {}%
  \def \@evenfoot {\uiobrev@info}%
  \let \@oddfoot = \@evenfoot}
\pagestyle{uiobrev}

%% The standard font is Times in the
%% version provided by the TeX Gyre project:

\RequirePackage{tgtermes}

%% And that was all.

\endinput
