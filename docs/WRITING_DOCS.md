# Writing the kika documentation

The rules for every page under `docs/source/`. They exist so that pages written by
different people, in different sessions, with different tools read as one manual. They
are not suggestions. A page that breaks one is not finished, however good it is
otherwise. To change a rule, edit this file first, in its own commit, then apply it.

The site is built by Read the Docs from `.readthedocs.yaml`
(<https://kika.readthedocs.io>). Pages are MyST Markdown (`.md`); the existing `.rst`
pages stay as they are until they are rewritten.

---

## 1. What the documentation is for

**The test for the whole manual: someone who has never read the code can answer "how does
kika compute this, under which assumptions, and how do we know it is right?" from the
pages alone.** This includes a referee, a user whose numbers disagree with NJOY's, and
the author six months from now.

The documentation answers questions about kika. It does not record how kika got there:
history, rejected alternatives, dates of decisions and work in progress belong in the
project's development notes, never on these pages.

## 2. The four sections, and where a paragraph goes

| Section | Directory | Answers | Changes when |
|---|---|---|---|
| **User guide** | `user_guide/` | *How do I do X with kika?* Task-oriented: a goal, the calls that reach it, what comes back. Tutorials are worked examples of the same thing. | The API changes |
| **Methods** | `methods/` | *How does kika compute X, and why that way?* The theory manual: equations, conventions, limits, validation. | The algorithm changes |
| **API reference** | `api_reference/` | *What exactly does this function take and return?* Generated from docstrings by autodoc. | A signature changes |
| **About** | `about/` | What kika is, licence, how to cite. | Rarely |

**The placement test, applied per paragraph, not per page:**

- It tells the reader which calls to make and in what order → **User guide.**
- It would stay true if the function were renamed, but not if the algorithm changed →
  **Methods.**
- It is the contract of one function (parameters, return value, exceptions) → the
  **docstring**, and therefore the API reference. Never restate it by hand on a page.
- It explains what a physical quantity *is* (what MF 34 asserts, what a Legendre
  expansion is) and is not specific to kika → **background**, the `Background` section of
  a methods page (§3). Only what the page needs to be read on its own, then a citation
  (the ENDF-6 manual, a textbook) for the full treatment.

A user-guide page never derives anything. It links to the methods page. A methods page
never walks through a workflow. It links to the user guide.

**The desktop app has its own guides**, which are not published with this site. The
library documentation does not document the app's panels, and never depends on the app's
guides: a reader of this site cannot follow a link into them. When a methods page and an
app guide explain the same thing, the methods page is the authority on what kika
computes, and the app guide follows it.

## 3. Methods pages: the template

Every methods page has these sections, **in this order, with these headings**. A section
that genuinely has nothing to say keeps its heading and one line saying so ("None: the
function never refuses an input."). Readers learn where to look; an absent heading reads
as a forgotten one.

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
{doc}`../notation`." when the notation page covers it.

## What it computes
Inputs → outputs, in words, then the defining equation of the result.

## Method
The algorithm, step by step, with every equation that defines the result. Numbered
display equations. A derivation longer than a few lines goes in a "### Derivation"
subsection at the end of this section.

## Conventions and choices
One bullet per decision a careful reader could have made differently: the choice, then
its reason in one or two sentences. "Why not X?" questions are answered here, briefly.

## Limits and failure modes
What it refuses (and the exception), what it reports instead of doing, and the known
limitations of the method itself.

## Validation
What the result is checked against (analytic case, NJOY, FUDGE, an independent path in
kika), the measured agreement, and the test that enforces it.

## Diagnostics
The keys a user finds in the returned `info`/report, what each means, and which ones
should worry them.

## Implementation
Which module does which step. No line numbers.

## References
````

A page that covers a family (e.g. the R-matrix formalisms) may repeat **What it computes**
→ **Limits** per member under h3 headings, but keeps one Validation, one Implementation
and one References section.

## 4. Level of detail

**The reproducibility bar.** A methods page is detailed enough that a competent reader
could reimplement the method from the page alone and land within the tolerances its
Validation section quotes. Below that bar it is a summary. Above it (every branch of the
code, every guard) it is a code listing in prose; that detail belongs in the code.

Concretely:

- **Every equation that defines the output is on the page.** Equations that only
  describe how the code is organised are not.
- **Every convention that changes a number is on the page**: frames, units, relative
  versus absolute, interpolation laws, how bin edges are treated, what happens at a
  threshold. "Implementation detail" is not an excuse for a convention that moves the
  result.
- **Every refusal is on the page.** If the code raises instead of guessing, the reader
  needs to know which inputs it refuses and why.
- **Length: one to four screens.** Longer means it is two methods. Split it and link the
  halves.

## 5. Content rules

### 5.1 Current state only

Present tense, describing what the code on the default branch does. Never on a page:

- history ("originally…", "was changed to…", "we first tried…");
- dates, except the date a validation number was measured;
- internal milestone or decision labels, branch names, commit hashes, issue numbers;
- paths to anything outside this repository, including private notes;
- results specific to one study, thesis or evaluation, unless it is the published
  example the page is built around and it is cited.

### 5.2 Numbers

- **Every validation number traces to something a reader can run**: a test in this repo
  (cite `path/to/test_file.py::test_name`) or a script under `examples/`. A number that
  only exists in a notebook or a private note does not go on a page.
- Give the comparison, the metric and the size: "maximum relative difference in σ(E)
  against NJOY 2016.76 RECONR, 0.1 % tolerance: 3·10⁻⁴ over 4 213 points". Never
  "agrees well".
- Two significant figures. Units always, with a space (`1.5 MeV`, `10.5 b`).
- State the date a measured number was taken, in the form `(measured 2026-10-10)`.

### 5.3 Notation

- One shared notation page, `methods/notation.md`. Every symbol used on more than one
  page is defined there, and pages use it exactly as defined. A new shared symbol goes
  into that page in the same change.
- A symbol used on only one page is defined on that page, at first use.
- Math syntax (MyST `dollarmath` and `amsmath`, enabled in `conf.py`): `$…$` inline;
  a display equation as `$$ … $$ (label)` on its own lines, which numbers it, and is
  referenced as ``{eq}`label` ``. Labels are `eq-<page-slug>-<name>`, so they stay
  unique across the site. Multi-line equations use `\begin{align} … \end{align}`.
- Fixed choices: `\sigma(E)` cross-section, `\mu = \cos\theta`, `a_\ell` Legendre
  coefficients with `f(\mu) = \sum_\ell \frac{2\ell+1}{2} a_\ell P_\ell(\mu)`, `E` incident
  energy in eV, CM/LAB for the frames, `\gamma` for the CM/LAB speed ratio.

### 5.4 Links and citations

- Cross-reference with MyST roles (`{doc}`, `{ref}`, `{py:func}`), never with a bare URL
  to this site. Every function named in prose links to its API entry the first time it
  appears on a page. If the function has no API entry yet, add it to the matching page
  under `api_reference/` in the same change; a role whose target does not exist renders
  as plain code, silently.
- Cite at the statement, not in a bucket at the end: the ENDF-6 manual by section
  ("ENDF-6 §34.1"), papers by author–year with the full entry under References. A
  formalism kika did not invent is cited where it is stated.
- Do not invent a reference. If a statement needs one that is not at hand, leave a
  visible `TODO(ref)` and say so when handing the page over.

### 5.5 Figures

SVG under `docs/source/_static/<section>/`, generated by a script committed next to it
(so the figure can be regenerated when the method changes). No screenshots of the app.
A figure carries a caption that says what to look at.

## 6. Prose and style

- **English, British spelling**, as in the code's docstrings: normalisation,
  linearised, parametrised.
- The kika-app guide conventions, so the two read alike:
  - **cross-section**, hyphenated; **covariance matrix**, two words;
  - nuclides as `U-235`, `Fe-56`, never `U235`;
  - in prose, `MF 4`, `MT 2` with a space; in code and identifiers as written (`MF4MT`);
  - expand on first use per page: ENDF (Evaluated Nuclear Data File), ACE, GNDS, NJOY.
- **Never write "rather than".** Use "instead of", "and not", or a full stop.
- Parentheses for asides. An em-dash only as a title separator, at most one per page
  otherwise.
- h1 is the title (one per page). h2 and h3 only below it; they form the sidebar, so make
  them descriptive.
- Say what a choice is *for* and *when it does nothing*. Do not restate what a name
  already says.
- No marketing ("powerful", "comprehensive", "seamless"). State what it does.
- **Do not guess.** Every claim is checked against the code and the tests before the page
  is handed over. A claim that could not be checked is either removed or marked
  `TODO(check)`.

## 7. Docstrings and pages

- The **docstring** is the contract of one function: parameters, return value,
  exceptions, and one paragraph saying what it does. Docstrings use NumPy style (they
  are rendered by `napoleon`).
- The **methods page** is the authority for the mathematics. A docstring that needs the
  method links to it (``See :doc:`/methods/<page>`.``) and does not re-derive it.
- Existing docstrings that already explain a method are not stripped in a documentation
  pass. When a methods page is written, the docstring's equations must agree with it;
  if they disagree, the code decides, and both are fixed.

## 8. Keeping it true

- **A change to an algorithm updates its methods page in the same commit.** A change to
  a public signature updates any user-guide page that calls it. The reviewer of a change
  checks this the same way they check the tests.
- A page names its tests in Validation. When one of those tests changes its tolerance or
  its measured number, the page changes with it.
- Pages are built with warnings as errors before they are handed over. Read the Docs
  builds the same way (`fail_on_warning` in `.readthedocs.yaml`), so a warning fails the
  published site too:

  ```bash
  # in a separate environment, in this order, as Read the Docs does
  pip install . -r docs-requirements.txt
  sphinx-build -W -b html docs/source docs/_build/html
  KIKA_DOCS_DRAFTS=1 sphinx-build -W -b html docs/source docs/_build/review
  ```

  The second build includes `drafts/` (§9); a draft page is only checked by it.

## 9. Procedure for writing a page

Every page, every author, the same four steps.

1. **Understand.** Read the code that implements the method, its tests, and any existing
   page or docstring on it. List the conventions and refusals you found in the code.
2. **Place.** Decide the section (§2) and the file name (`methods/<chapter>/<slug>.md`,
   lower-case, hyphenated). Check no other page already covers it. Name the notation
   it needs and whether it is in `notation.md`.
3. **Write** to the template (§3), at the level of §4, under the rules of §5-§6.
4. **Verify.** Re-read every claim against the code. Trace every number to a test.
   Build with `-W`. Then hand over with a list of what was checked, what is marked
   `TODO`, and what you were unsure about.

A page is **published** (wired into a toctree) only after the maintainer has read it.
Until then it lives under `docs/source/drafts/`. The published build leaves `drafts/`
out. A **review build** includes it, under a "Drafts (under review)" section and with a
banner on every draft page: `KIKA_DOCS_DRAFTS=1` locally, and on Read the Docs every
pull-request preview and the `develop` version (`docs/source/conf.py`). To publish a
page, move it out of `drafts/` to its place and add it to its section's toctree.
