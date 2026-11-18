# Writing the kika manual

The rules for the **kika manual**: the technical manual of how kika is built, how it reads
and writes each format, and how it computes what it computes. It is published as a PDF
and, page for page, in the Architecture and Methods sections of
<https://kika.readthedocs.io>. Pages written by different people, in different sessions,
with different tools, must read as one book. These rules are not suggestions. A page that
breaks one is not finished, however good it is otherwise. To change a rule, edit this file
first, in its own commit, then apply it.

The rest of the documentation (the user guide, the API reference, the site itself) follows
`WRITING_DOCS.md`. Four parts of this file apply to every page of the site: what never
goes on a page (§1.2), numbers and units (§7), citations and links (§9) and prose (§12).

---

## 1. What the manual is

**The test for the whole manual: someone who has never read the code can answer "how is
kika built, how does it read and represent the data, how does it compute this, under which
assumptions, and how do we know it is right?" from the manual alone.** Its readers are a
referee, an evaluator whose numbers disagree with NJOY's, a contributor about to add a
format, and the author six months from now. It stands next to the manuals of NJOY, MCNP
and SCALE, and is held to their level: every method with its theory, its choices, its
limits and its evidence.

**It is 100 % technical.** It explains what kika computes and why. How to use kika is the
user guide's job; what one function takes and returns is the API reference's. The manual
links to them and never repeats them.

**Design decisions are documented with their reasons.** Every choice (the layers, the
canonical model, how a format is read, what is refused, which algorithm) is stated as the
decision, the argument for it, and what follows from it. An alternative is named to say,
in the present tense, why kika does not take it.

**Current state only.** The manual describes what the code on the default branch does
(§1.2).

**It is the authority.** When a docstring, a user-guide page or a guide of the desktop app
explains the same thing, the manual is right and the others follow it. The manual never
depends on the app's guides: a reader of the PDF cannot follow a link into them.

### 1.1 What goes in the manual

Applied per paragraph, not per page:

- It would stay true if an algorithm changed, but not if the layers, the model or the way
  formats reach it changed → **Architecture.**
- It says what kika reads from a file, what it assumes where the file is silent, what it
  refuses, or what is lost writing it back → **Formats.**
- It would stay true if the function were renamed, but not if the algorithm changed →
  **Methods.**
- It explains what a physical quantity *is* (what MF 34 asserts, what a Legendre
  expansion is) and is not specific to kika → the **Background** section of the page that
  needs it: only what the page needs to be read on its own, then a citation (the ENDF-6
  manual, a textbook) for the full treatment.
- It tells the reader which calls to make and in what order → not the manual: the
  **user guide** (`WRITING_DOCS.md`).
- It is the contract of one function → not the manual: the **docstring**.

### 1.2 Never on a page

Present tense, describing what the code on the default branch does. History, dates of
decisions and the order things were built in belong in the project's development notes.
Never on any page of the site, manual or not:

- history ("originally…", "was changed to…", "we first tried…");
- dates, except the date a validation number was measured (§7) and the
  `versionchanged` line of §10;
- internal milestone or decision labels, branch names, commit hashes, issue numbers;
- paths to anything outside this repository, including private notes;
- results specific to one study, thesis or evaluation, unless it is the published
  example the page is built around and it is cited.

## 2. The book

### 2.1 Its parts

The PDF is built from `docs/source/pdf_index.rst`, which fixes the reading order. The
online site shows the same pages under Architecture and Methods.

| Part | Directory | Contents |
|---|---|---|
| Front matter | `about/` | Preface; what kika is; licence; how to cite |
| I. Architecture | `architecture/` | The layers, the canonical model, the format adapters, the algebra layer |
| II. Formats | `methods/formats/` | One chapter per format: how kika reads it into the model and writes it back |
| III. Methods | `methods/<chapter>/` | One chapter per field (algebra, processing, covariance, perturbation, sampling, uncertainty, checks), one page per method |
| Appendices | `methods/`, `about/` | Notation; glossary; known issues |

### 2.2 Front matter and chapter openings

- **The preface** (`about/preface.md`) says who the manual is for, what it covers and what
  it does not (the user guide and the API are online), how to read it (Part I first; then
  the format or method needed, each chapter stands on its own after the notation
  appendix), and the conventions every page uses (units, frames, the notation appendix).
- **Every part and every chapter opens with an index page**: two or three paragraphs on
  what the chapter covers, what it leaves to other chapters, and the order of its pages
  when one builds on another; then one line per page. A list of links with nothing around
  it is not a chapter opening.
- **One page is one method** (or one format, or one structural element). A page that turns
  out to cover two is split. Length alone is never a reason to split.

### 2.3 Read as a book, browsed as a site

- A page is read in the PDF with no link to follow. Every sentence must make sense
  without its link: write "the user guide shows how to run it ({doc}`…`)", never "see
  {doc}`here <…>`".
- Links to the user guide and the API become plain text in the PDF. They are fine, as long
  as the sentence carries the meaning.
- Figures, tables and display equations are numbered, and the text refers to them by
  number ("Figure 2 shows…", "{eq}`eq-doppler-kernel`"), never by position ("the figure
  below").

## 3. Page templates

Methods pages, format pages and architecture pages each have a fixed template. Every page
has its sections **in this order, with these headings**. A section that genuinely has
nothing to say keeps its heading and one line saying so ("None: the function never refuses
an input."). Readers learn where to look; an absent heading reads as a forgotten one.

### 3.1 Methods pages

````markdown
# <What it does, as a gerund phrase>          ← "Perturbing a tabulated angular distribution"

```{admonition} At a glance
:class: note
**Computes:** one sentence.
**Applies to:** formats, MF/MT, model nodes.
**Entry points:** `module.function`, `module.Class.method`.
**Status:** stable | experimental — since <version>.
```

## Background
The physics and format facts the page relies on, at the minimum needed to read it
without another document, with citations. Usually one to three paragraphs; "None beyond
{doc}`../notation`." when the notation appendix covers it.

## What it computes
Inputs → outputs, in words, then the defining equation of the result.

## Method
The algorithm, step by step, with every equation that defines the result. Numbered
display equations. A derivation longer than a few lines goes in a "### Derivation"
subsection at the end of this section. A worked example, when there is one (§6), goes in
a "### Worked example" subsection after it.

## Conventions and choices
One bullet per decision a careful reader could have made differently (§3.4).

## Limits and failure modes
What it refuses (and the exception), what it reports when it does not compute, and the known
limitations of the method itself, with their size when it is known.

## Validation
What the result is checked against (analytic case, NJOY, FUDGE, an independent path in
kika), the measured agreement, and the test that enforces it (§7).

## Diagnostics
The keys a user finds in the returned `info`/report, what each means, which ones should
worry them, and what to do when they do.

## Implementation
Which module does which step. No line numbers.

## References
````

A page that covers a family (e.g. the R-matrix formalisms) may repeat **What it computes**
→ **Limits** per member under h3 headings, but keeps one Validation, one Implementation
and one References section.

### 3.2 Format pages (`methods/formats/`)

One page per format (ENDF, GNDS, ACE, G4NDL, HDF5), on how kika reads it into the
canonical model and writes the model back. A format with much to say may split reading
and writing into two pages; both keep the template.

````markdown
# Reading and writing <format>

```{admonition} At a glance
:class: note
**Reads:** what kika decodes, in one sentence.
**Writes:** what it encodes.
**Into:** the model objects it produces (link the architecture page).
**Entry points:** `module.function`.
**Status:** stable | experimental — since <version>.
```

## Background
The format at the minimum needed to read the page: its units, its records, how its data
is organised, with citations to the format's own manual.

## Coverage
A table: each part of the format (section, record type, node) → what kika does with it
(decoded into which model object, kept verbatim, or refused), with the reason in a few
words when it is not decoded.

## Reading
How the file becomes the model, step by step: the grammar rules that change a number
(how a float is parsed, how interpolation regions map to model functions, how a step is
spelled), and which model object each part lands in.

## Assumptions and defaults
What kika assumes where the file is silent (a default value, a unit, a frame), one bullet
each, with its reason.

## Ambiguous and inconsistent input
Each case where the file allows more than one reading, or contradicts itself: what kika
does, why, and what it reports.

## Refused
What kika will not read, the exception it raises, and why it raises and does not guess.

## Writing
How the model is written back: what is derived because the model does not carry it,
the formatting rules, what is recomputed (counts, pointers, sums).

## What the round trip keeps and loses
Read → model → write: what comes back bit for bit, what comes back numerically equal,
what is not preserved, and why.

## Report
The entries a user finds in the conversion report, what each means, and which ones
should worry them.

## Validation
The round-trip and oracle checks (another reader, FUDGE, NJOY), on which files, with the
test that enforces each.

## Implementation
Which module does which step. No line numbers.

## References
````

### 3.3 Architecture pages (`architecture/`)

On the structure of kika: an overview that maps the whole, and one page per structural
element.

````markdown
# <The element, as a noun phrase>           ← "The canonical model"

```{admonition} At a glance
:class: note
**Decides:** the structural rule, in one sentence.
**Packages:** the packages it covers.
**Enforced by:** the test or mechanism that holds it, or "convention only".
```

## The structure
What exists and how the parts connect, with a figure when the connections are more than
a list.

## Design decisions
One h3 per decision: the decision, why kika makes it, and what follows from it (what it
costs, what it makes possible). The heart of the page.

## What it rules out
What code may not do under this structure, and how a violation is caught.

## Known exceptions
Every place the code does not yet follow the structure, why it stands, and what keeps it
from growing. "None." when there are none.

## Extending it
Where new work goes: which layer a new format, quantity or algorithm belongs to, and the
test it has to pass.

## Implementation
Which module holds which part. No line numbers.

## References
````

The overview page is the exception: it keeps **The structure** and **Design decisions**,
then a short section per element linking its page, and no **Known exceptions** of its own.

### 3.4 Conventions and choices: how a decision is argued

This section is what makes the manual worth reading next to NJOY's. Each bullet (or h3,
when the argument is long):

1. **The choice**, in one sentence.
2. **The argument**: what is at stake (accuracy, cost, physics, robustness), and why this
   choice wins on it. When the argument is a trade-off, state both sides before the
   choice, as NJOY's RECONR chapter does for resonance thinning.
3. **What the other codes do.** Where kika does something other than NJOY, FUDGE, FRENDY,
   OpenMC or SCALE, say what they do, why kika does not, and what the difference buys,
   with the evidence (§7). When it is only a different route to the same result, say so:
   equivalence is a valid outcome, and "better" is claimed only where it is measured.
4. **When it does nothing.** If the choice only matters in some cases (a threshold, a
   format flag, one temperature range), say which, so a reader whose result does not move
   knows that is expected.

## 4. Level of detail

**The reproducibility bar.** A methods page is detailed enough that a competent reader
could reimplement the method from the page alone and land within the tolerances its
Validation section quotes. A format page is detailed enough that a competent reader could
write a reader producing the same model from the same file, and predict what kika does
with any section of it. An architecture page is detailed enough that a reader can say,
for any piece of code, which layer it belongs to and which imports it may not make. Below
that bar it is a summary. Above it (every branch of the code, every guard) it is a code
listing in prose; that detail belongs in the code.

- **Every equation that defines the output is on the page.** Equations that only describe
  how the code is organised are not.
- **Every convention that changes a number is on the page**: frames, units, relative
  versus absolute, interpolation laws, how bin edges are treated, what happens at a
  threshold. "Implementation detail" is not an excuse for a convention that moves the
  result.
- **Every refusal is on the page.** If the code raises and does not guess, the reader needs
  to know which inputs it refuses and why.
- **No length limit.** A page is as long as its subject needs.

## 5. Figures

The NJOY manual is read for its figures as much as for its equations: Doppler broadening
shown on a real resonance at three temperatures, a multigroup set drawn over the
pointwise curve. The kika manual holds the same standard, and adds what NJOY lacks: every
figure can be regenerated.

### 5.1 When a page has a figure

A figure is there when it shows what prose cannot: the shape of a curve, a comparison, a
mesh, a geometry, a flow of data.

- **A methods page whose result is a function** (of energy, angle, or both) shows that
  result on a real evaluation, at the scale where the method matters (a resonance
  broadened, a distribution tabulated, a grid refined), next to its input or its
  reference.
- **A page that builds a grid, a ladder or a mesh**, or an architecture page whose parts
  connect in more than a list, has a schematic.
- **A Validation section with more than a handful of numbers** shows the difference
  against the reference as a plot, with the tolerance drawn on it.
- A figure that shows nothing beyond what its paragraph already makes plain is left out.
  A figure is never an image of code, of a terminal, of a table or of the app.

### 5.2 How a figure reads

- It is introduced in the text before it appears and referred to by its number.
- Its caption says **what to look at** ("the 600 K curve fills the dips between
  resonances; the area under each resonance is kept"), not only what is drawn.
- Every conclusion it shows is also stated in the text, so the page still reads without it.
- A plot of data names the nuclide, the evaluation and its version, the temperature, and
  the quantity and units on both axes. Lines are told apart by style as well as by colour,
  so the figure survives a black-and-white print.

### 5.3 How a figure is made

SVG under `docs/source/_static/<section>/`, generated by a script committed next to it, so
the figure is regenerated when the method changes. The script reads its data from the
small evaluations the tests use, or from a library named in its first lines, and writes
the figure as SVG and as PDF. The page cites it as `name.*`: Sphinx takes the SVG for the
web and the PDF for the LaTeX build, which cannot include an SVG.

## 6. Worked examples and output examples

- **A methods page may carry one worked example** (an h3 "Worked example" at the end of
  *Method*) when a convention changes a number and a reader is likely to get it wrong: a
  few input values, the result step by step, and the test that holds the same values.
- **Diagnostics may show an excerpt of the report** the method produces (a few lines, as
  text), with each entry explained. This is NJOY's "reading the output", and it is the
  one place a methods page shows output.
- The manual has no tutorials. How to run a method is the user guide's job; the page links
  to it. When the user guide is rewritten around tasks, a short chapter of its worked
  examples may join the PDF as an appendix; it is one line in `pdf_index.rst`.

## 7. Numbers and validation

- **Every validation number of kika against its own physics traces to something a reader
  can run**: a test in this repo (cite `path/to/test_file.py::test_name`) or a script under
  `examples/`. This is the evidence that kika computes what the page says.
- **A comparison with another code** (NJOY, FUDGE, FRENDY, OpenMC, SCALE) may be quoted
  without a test in this repo, because those codes are not dependencies of kika and are
  not the truth: a test pinned to their output would fail the day kika improves on it. The
  page states the code and its version, the evaluation and material, the tolerances of
  both sides, the metric and the date measured. When the two disagree, the page says which
  one an independent reference supports, or that it is not known.
- Give the comparison, the metric and the size: "maximum relative difference in σ(E)
  against NJOY 2016.76 RECONR, 0.1 % tolerance: 3·10⁻⁴ over 4 213 points". Never
  "agrees well".
- Two significant figures. Units always, with a space (`1.5 MeV`, `10.5 b`).
- State the date a measured number was taken, in the form `(measured 2026-10-10)`.
- **A limitation has a size when it has one**: "underestimates <quantity> by up to
  <size> below <energy>" tells a reader whether it matters to them; "may be less
  accurate" does not.

## 8. Notation and equations

- One shared notation appendix, `methods/notation.md`. Every symbol used on more than one
  page is defined there, and pages use it exactly as defined. A new shared symbol goes
  into that page in the same change.
- A symbol used on only one page is defined on that page, at first use.
- Math syntax (MyST `dollarmath` and `amsmath`, enabled in `conf.py`): `$…$` inline;
  a display equation as `$$ … $$ (label)` on its own lines, which numbers it, and is
  referenced as ``{eq}`label` ``. Labels are `eq-<page-slug>-<name>`, so they stay unique
  across the book. Multi-line equations use `\begin{aligned} … \end{aligned}` inside the
  `$$` block (a bare `align` there renders on the web and breaks the PDF build).
- Fixed choices: `\sigma(E)` cross-section, `\mu = \cos\theta`, `a_\ell` Legendre
  coefficients with `f(\mu) = \sum_\ell \frac{2\ell+1}{2} a_\ell P_\ell(\mu)`, `E` incident
  energy in eV, CM/LAB for the frames, `\gamma` for the CM/LAB speed ratio.
- Terms used across chapters (evaluation, tape, style, suite, …) are defined once in the
  glossary appendix (`about/glossary.md`) and linked with `{term}` at first use on a page.

## 9. Citations, links and code

- Cross-reference with MyST roles (`{doc}`, `{ref}`, `{py:func}`), never with a bare URL to
  this site. Every function named in prose links to its API entry the first time it
  appears on a page. If the function has no API entry yet, add it under `api_reference/`
  in the same change; a role whose target does not exist renders as plain code, silently.
- Cite at the statement, not in a bucket at the end: the ENDF-6 manual by section
  ("ENDF-6 §34.1"), papers by author–year with the full entry under References. A
  formalism kika did not invent is cited where it is stated.
- Do not invent a reference. If a statement needs one that is not at hand, leave a
  visible `TODO(ref)` and say so when handing the page over.
- **No implementation code.** The method is stated in words and equations; the
  *Implementation* section names the module that does each step, and a reader who wants
  the code opens it. A code block appears only when it is the subject: a line of a
  format's records, the exact text of a field, a report excerpt (§6). Never a function
  body, a class definition or a call sequence.
- Inline code names things (a function, a module, a key of a report, an MF number as the
  file spells it). It is not a substitute for saying what the thing does.

## 10. Changed results and known issues

The pages describe the current state. A change that alters the numbers kika produces, or
a defect that is known and not yet fixed, is recorded in places that are themselves
current:

- **`CHANGELOG.md`** at the repository root, per release, under *Changed results*: which
  functions, which data, the size of the change on a named case, and why. Every fix says
  its effect on results, including "none".
- **The known-issues appendix** (`about/known-issues.md`): one entry per open defect, or
  per fixed defect whose wrong results a user may still hold. Each entry: the symptom,
  the affected versions and data, the workaround, and the version that fixes it. An entry
  is removed when no supported version is affected.
- **The page of the method** carries one ```` ```{versionchanged} <version> ```` line,
  saying what the method does now and linking the changelog entry. With the date of a
  measurement (§7) and the version in *At a glance*, it is the only dated statement on a
  page. It does not tell the story.

A known limitation of the method itself (not a defect) stays in *Limits and failure
modes* and is not a known issue.

## 11. Keeping it true

- **A change to an algorithm updates its methods page in the same commit.** The reviewer
  of a change checks this the same way they check the tests.
- A page names its tests in Validation. When one of those tests changes its tolerance or
  its measured number, the page changes with it. When a figure's script or data changes,
  the figure is regenerated in the same commit.
- Pages are built with warnings as errors before they are handed over, as
  `WRITING_DOCS.md` §7 describes. The book is checked too, because a page that builds on
  the web can still break LaTeX:

  ```bash
  KIKA_DOCS_DRAFTS=1 sphinx-build -W -b latex docs/source docs/_build/latex
  cd docs/_build/latex && latexmk -xelatex -interaction=nonstopmode kika.tex
  ```

  The first command catches Sphinx's problems; the second, LaTeX's (an unsupported
  character, an `align` outside `aligned`, a figure without its PDF twin).

## 12. Prose and style

These rules apply to the whole documentation, the user guide included.

- Units always, with a space (`1.5 MeV`, `10.5 b`); numbers to two significant figures
  unless more are the point.
- **English, British spelling**, as in the code's docstrings: normalisation, linearised,
  parametrised.
- The kika-app guide conventions, so the two read alike:
  - **cross-section**, hyphenated; **covariance matrix**, two words;
  - nuclides as `U-235`, `Fe-56`, never `U235`;
  - in prose, `MF 4`, `MT 2` with a space; in code and identifiers as written (`MF4MT`);
  - expand on first use per page: ENDF (Evaluated Nuclear Data File), ACE, GNDS, NJOY.
- **Never write "rather than".** Use "instead of" sparingly: most contrasts read better
  stated in the positive (say what the thing is), or split with a full stop. More than
  one "instead of" in a paragraph means at least one should go.
- Parentheses for asides. An em-dash only as a title separator, at most one per page
  otherwise.
- h1 is the title (one per page). h2 and h3 only below it; they form the sidebar and the
  PDF's table of contents, so make them descriptive.
- Say what a choice is *for* and *when it does nothing*. Do not restate what a name
  already says.
- No marketing ("powerful", "comprehensive", "seamless"). State what it does.
- **Do not guess.** Every claim is checked against the code and the tests before the page
  is handed over. A claim that could not be checked is either removed or marked
  `TODO(check)`.

## 13. Procedure for writing a page

Every page, every author, the same four steps.

1. **Understand.** Read the code that implements the method, its tests, and any existing
   page or docstring on it. List the conventions and refusals you found in the code, and
   what NJOY or the other codes do for the same step.
2. **Place.** Decide the part and chapter (§1.1, §2.1) and the file name
   (`methods/<chapter>/<slug>.md`, `methods/formats/<format>.md` or
   `architecture/<slug>.md`, lower-case, hyphenated). Check no other page already covers
   it. Name the notation it needs and whether it is in the notation appendix. Name the
   figure it needs (§5.1).
3. **Write** to the template of its kind (§3), at the level of §4, under the rules of
   §5-§12. Write the figure's script with the page.
4. **Verify.** Re-read every claim against the code. Trace every number to a test or a
   dated comparison. Build the web and the PDF with `-W`. Then hand over with a list of
   what was checked, what is marked `TODO`, and what you were unsure about.

A page is **published** only after the maintainer has read it. Until then it lives under
`docs/source/drafts/` (`WRITING_DOCS.md` §8). To publish it, move it to its place and add
it to the toctrees of both `index.rst` (the site) and `pdf_index.rst` (the book), in the
chapter and position §2.1 gives it.
