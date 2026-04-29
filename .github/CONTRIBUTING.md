# Contributing guidelines

Thank you for you interest in contributing to this project! When drafting you PR, please follow these guidelines.

## Developer Environment

- [Install Pixi](https://pixi.prefix.dev/latest/installation/) if you don't have it already:

        # on Linux or MacOS
        curl -fsSL https://pixi.sh/install.sh | bash

- Fork the repository on GitHub
- Clone the repository locally, add upstream remote and fetch latest

        git clone https://github.com/<your-username>/firecode.git
        git remote add upstream https://github.com/ntampellini/firecode.git
        git fetch upstream

- Make the desired changes/additions (see guidelines below)
- Create minimal tests in `test_suite.py` that cover the new feature / bug if possible
- If the new feature adds a dependency, the standard practice is to add it to the `dev` environment only. Of course, this is also why testing is done in the `dev` environment

        pixi add --feature dev new_dependency

- Commits are protected by a pre-commit hook, letting you commit only if formatting tests are passed. You can run the hook without committing with:

        pixi run -e dev pre-commit

- The CI test suite is not invoked in the pre-commit hook, since tests take ~2 minutes. You can run it with:

        pixi run -e dev test_cov

- When all tests and pre-commit hooks pass, your PR is ready for review!

## Code guidelines

- **Styling**: the code is styled with `ruff` and type-checked with `mypy` by the `pre-commit` hook. As long as the hook passes, the code style/linting/typing should be good to go.

- **Comments**: the code should be [as pythonic as possible](https://realpython.com/ref/best-practices/pythonic-code/) to make it easier to interpret. When not obvious, comment strings should be used to clarify the general idea behind the code block.

### Code organization

#### Adding a new `operator>`

The addition of new operators is straightforward by design: just create a new function with the following arguments and returning the filename of output structures. The run data can be accessed by `embedder` or its `embedder.options`.

```python
def center_operator(filename: str, embedder: Embedder) -> str:
    """Example operator centering the molecule."""

    # the Embedder class stores global information
    embedder.avail_gpus # 1
    embedder.options.solvent # "ch2cl2"
    embedder.options.T # 298.15

    # get a copy of the molecule of interest
    mol = embedder.mols[filename]

    # center coordinates
    mol.coords[0] -= np.mean(mol.coords[0], axis=0)

    # save any data you might need later
    embedder.options.centered = True
    embedder.options.last_operator = "center"

    # write to global log
    embedder.log(f"--> Center operator: centered molecule {mol.basename}")

    # write outfile and return its name
    outfile = f"{mol.basename}_centered.xyz"
    mol.to_xyz(outfile)

    return outfile
```

This operator function will be called from `operate`, where you just have to add a `case` with the desired operator name(s):

```python
def operate(filename: str, operator: str, embedder: Embedder) -> str:
    """ [...] """

    [...]

    match operator:
        case "opt":
            outname = opt_operator(filename, embedder)

        [...]

        # this is our new operator
        case "center" | "centre":
            outname = center_operator(filename, embedder)

        [...]

```
#### Adding a new keyword

Keywords are processed in `embedder_options.py`. Just add the new keyword to `keywords_dict` and write a function with the same name in the `Option_setter` class, following the syntax of other keywords. See the `CHARGE` keyword as an example:

```python
def charge(self, options: Options, *args: Any) -> None:
    kw = self.kw_table["CHARGE"]
    options.charge = int(kw.split("=")[1])
```

#### Adding a new calculator interface

The easiest way to interface a new calculator is via ASE. The general interface for all calculators is the `Dispatcher` class in `dispatcher.py`. New calculators should also get their specific environment variables in `settings.py`.
