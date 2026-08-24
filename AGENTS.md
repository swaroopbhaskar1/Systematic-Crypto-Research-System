# AGENTS.md

## Cursor Cloud specific instructions

### Current repository state (important)

As of this writing, this repository is effectively **empty / greenfield**. It
contains only `README.md`, `LICENSE`, and a Java-oriented `.gitignore`. There is
**no application code, no dependency manifest (e.g. `pom.xml`, `build.gradle`,
`package.json`, `requirements.txt`), and no build system**. Consequently there
is nothing to build, run, lint, or test yet, and the update script is
intentionally a safe no-op until real project files are added.

### Available toolchains (preinstalled in the Cloud VM)

These are provided by the base image; the update script does not install them:

- Java: OpenJDK **21** (`java`, `javac`). This matches the Java-focused
  `.gitignore` and is the most likely intended stack.
- Node.js **22** (`node`, `npm`).
- Python **3.12** (`python3`, `pip3`).
- Go **1.22**, Rust (`rustc`).

Not preinstalled: **Maven (`mvn`) and Gradle are NOT available.** If/when a Java
build is added, install the matching build tool (or vendor a wrapper such as
`./mvnw` / `./gradlew`) as part of the update script.

### When project files are added, wire them up here

Update the startup update script (Cursor Cloud "environment" update script) to
install dependencies for whatever manifest lands, for example:

- Maven: `mvn -q -DskipTests dependency:go-offline` (requires installing Maven first).
- Gradle: `./gradlew --no-daemon dependencies` (prefer the committed wrapper).
- Node: `npm ci` (or `npm install` if there is no lockfile).
- Python: `pip3 install -r requirements.txt` (prefer a virtualenv).

Keep the update script idempotent and guarded so it stays valid even before the
corresponding manifest exists on a given branch.
