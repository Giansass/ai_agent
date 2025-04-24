# A Python project template
<a id="readme-top"></a>
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Balck][black-shield]][black-url]
[![Precommit][pre-commit-shield]][pre-commit-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">python-project-template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Programming language used</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to define a template to use to develop a Python project. The template prepares the basic Python environment necessary to proceed with development and makes use of the main libraries useful for writing correct Python code.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Programming language used

This section lists the programming languages used in the project:

[![Python][Python.com]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

To use this templete assure:

1. Python is installed. [Here](https://realpython.com/installing-python/) a user friendly guide to install python on Windows, macOS and Linux.
2. Poetry is installed. [Here](https://python-poetry.org/docs/#installing-with-the-official-installer) the offical guide to install Poetry.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

In order to properly install the template take the following steps:

1. Use this template to [create a new repository from template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) in GitHub.
2. Clone the created repository on your local system
   ```sh
   git clone https://github.com/<user_name>/<repo_name>.git
   ```
3. Change info in the project in the pyproject.toml
   ```sh
   [tool.poetry]
   name = "<name_of_the_project>"
   version = "0.1.0"
   description = "<project_description>"
   authors = ["author_name"]
   license = "<license>"
   readme = "README.md"
   ```
3. Create a poetry virtualenv
   ```sh
   poetry env use python3
   ```
4. Create the poetry.lock form the pyproject.toml (I decided to not committing the poetry.lock since the project doesn't not need to bee perfectly reproducible in other environments, here the [official documentation](https://python-poetry.org/docs/basic-usage/#committing-your-poetrylock-file-to-version-control))
   ```sh
   poetry lock
   ```
5. Install all the packages in the poetry.lock
   ```sh
   poetry install
   ```

6. Activate the created virtualenv
   ```sh
   source $(poetry env info --path)/bin/activate
   ```
   
7. Install all pre-commit hooks
   ```sh
   pre-commit install --hook-type pre-commit --hook-type pre-push
   ```

8. Modify project, copyright, author and release from the docs/conf.py used by Sphinx
   ```sh
   project_name=$(git config --local remote.origin.url|sed -n 's#.*/\([^.]*\)\.git#\1#p')
   sed -i "" "s/python_project_template/${project_name}/" docs/conf.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] README updated
- [x] Add pytest in pre-push step
- [x] Add Sphinx initialization
- []

See the [open issues](https://github.com/Giansass/python-project-template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repository and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors

<a href="https://github.com/Giansass/python-project-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Giansass/python-project-template" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [LICENSE](https://github.com/Giansass/python-project-template/blob/develop/LICENSE) for more information.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Andrea Giansanti
* Email: <a href="mailto:andreagiansanti90@gmail.com">andreagiansanti90@gmail.com</a>
* Linkedin: <a href="https://www.linkedin.com/in/andreagiansanti/">https://www.linkedin.com/in/andreagiansanti/</a>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Giansass/python-project-template.svg?style=for-the-badge
[contributors-url]: https://github.com/Giansass/python-project-template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Giansass/python-project-template.svg?style=for-the-badge
[forks-url]: https://github.com/Giansass/python-project-template/network/members
[stars-shield]: https://img.shields.io/github/stars/Giansass/python-project-template.svg?style=for-the-badge
[stars-url]: https://github.com/Giansass/python-project-template/stargazers
[issues-shield]: https://img.shields.io/github/issues/Giansass/python-project-template.svg?style=for-the-badge
[issues-url]: https://github.com/Giansass/python-project-template/issues
[license-shield]: https://img.shields.io/github/license/Giansass/python-project-template.svg?style=for-the-badge
[license-url]: https://github.com/Giansass/python-project-template/blob/develop/LICENSE
[Python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[pre-commit-shield]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=for-the-badge
[pre-commit-url]: https://pre-commit.com/
[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[black-url]:https://github.com/psf/black
