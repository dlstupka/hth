# Toolchain Configuration

## Cross-platform execution model

The regression workflow minimizes operating-system-specific logic.

### Python provisioning

-   **Linux / GitHub-hosted runners** use `actions/setup-python`.
-   **Windows self-hosted runners** use a verified Python installation
    in the GitHub Actions tool cache (`_work/_tool/Python/...`) rather
    than reinstalling Python on every run.

This avoids repeated Windows Installer/UAC interactions while keeping
the benchmark environment deterministic.

## Why Git Bash instead of WSL?

The Windows self-hosted runner intentionally executes all cross-platform
pipeline steps under **Git for Windows (MSYS2) Bash**, not WSL.

Reasons:

-   A single native Windows Python interpreter is used for every
    regression.
-   Git Bash provides POSIX shell semantics while executing the native
    Windows interpreter.
-   WSL introduces a second operating system, separate Python
    environment, `/mnt/...` path translation, and different filesystem
    semantics.
-   Using Git Bash keeps Windows behavior closely aligned with
    GitHub-hosted Linux while preserving native Windows performance.

## PATH precedence

On the Windows runner:

1.  `C:\Program Files\Git\bin\bash.exe`
2.  `C:\Windows\System32\bash.exe` (WSL launcher)
3.  Windows App aliases

This ensures `shell: bash` resolves to Git Bash.

## Toolchain diagnostics

Every regression records:

-   Runner name, OS, architecture and tool cache
-   SHELL, MSYSTEM and OSTYPE
-   Bash, sh, Git, Python and pip resolution
-   `sys.executable`, Python version and platform
-   Windows `where.exe` output for bash, sh, git, python and py
-   PATH contents
-   Git Bash filesystem translation:
    -   `pwd`
    -   `pwd -W`
    -   `cygpath -w .`
    -   `cygpath -u "$GITHUB_WORKSPACE"`

These diagnostics make environment drift immediately visible.

------------------------------------------------------------------------

# Lessons Learned

## 1. Separate infrastructure failures from application failures

The original regression failure was not caused by the detector. It was
caused by workflow infrastructure. Fixing infrastructure first
dramatically reduced debugging time.

## 2. Self-hosted Windows is not GitHub-hosted Windows

`actions/setup-python` interacts with the Windows Installer database. On
a self-hosted runner, protected registry cleanup can fail, leaving a
partially provisioned tool cache.

The workflow now verifies an existing Python installation instead of
reprovisioning it on every run.

## 3. Shell selection matters

`shell: bash` is only deterministic if Git Bash appears ahead of the WSL
launcher on PATH.

Using WSL accidentally changes filesystem semantics and can execute an
entirely different Python environment.

## 4. One interpreter, one benchmark

Benchmarking is only meaningful when every run uses the same Python
interpreter and OpenCV installation.

The workflow intentionally executes Git Bash while continuing to use the
native Windows Python tool cache.

## 5. Capture the environment

A regression log should explain not only *what* happened, but *where* it
happened.

Recording shell resolution, executable resolution, PATH ordering and
filesystem translation has already proven valuable and is expected to
remain part of the regression artifact set.
