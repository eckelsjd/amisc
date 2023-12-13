## Contributing to `amisc`
You might be here if you want to:

- Report a bug
- Discuss the current state of the code
- Submit a fix
- Propose a new feature
- Write unit tests
- Add to the documentation

We use [Github](https://guides.github.com/introduction/flow/index.html) to host code and documentation, to track issues and feature requests, and to accept pull requests.

## Submitting pull requests
Pull requests are the best way to propose changes to the codebase (bug fixes, new features, docs, etc.)

1. Fork the repo and create your branch from `main`. 
3. If you are adding a feature or making major changes, first create the [issue](https://github.com/eckelsjd/amisc/issues). 
4. If you've added code that should be tested, add to `/tests`. 
5. If you've made major changes, update the `/docs`. 
6. Ensure the test suite passes (`pdm run test`).
7. Make sure your code passes lint checks (coming soon). 
8. Follow [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) guidelines when adding a commit message.
9. Issue that pull request!

We strongly recommend using [pdm](https://github.com/pdm-project/pdm) to set up your development environment. An example contribution workflow is shown here:

```shell
pip install --user pdm

# Fork the repo on Github

git clone https://github.com/<your-user-name>/amisc.git
cd amisc
pdm install
git checkout -b <your-branch-name>

# Make local changes

pdm run test  # make sure tests pass
git add -A
git commit -m "Adding a bugfix or new feature"
git push -u origin <your-branch-name>

# Go to Github and "Compare & Pull Request" on your fork
# For your PR to be merged:
  # squash all your commits on your branch (interactively in an IDE most likely)
  # rebase to the top of origin/main to include new changes from others
  
git fetch
git rebase -i main your-branch  # for example
  
# Resolve any conflicts
# Your history now looks something like this:
#              o your-branch
#             /
# ---o---o---o main
  
# You can delete the branch and fork when your PR has been merged
```

You can also find a good tutorial [here](https://github.com/firstcontributions/first-contributions/tree/main).

## Report bugs using [issues](https://github.com/eckelsjd/amisc/issues)
Open a new issue and describe your problem using the template. Provide screenshots where possible and example log files.
Add labels to help categorize and describe your issue.

## Community
Start or take part in community [discussions](https://github.com/eckelsjd/amisc/discussions) for non-code related things.

## License
By contributing, you agree that your contributions will be licensed under its GNU GPLv3 License.

## Releases
The package version is tracked at `amisc.__init__.__version__`. You should not edit this value. The version will be 
increased on a case-by-case basis and released depending on the changes being merged.